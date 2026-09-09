"""CPU-only planning for bounded, binary-width target-verify CUDA graphs.

Candidate lengths are never changed by this planner. Execution lengths may
include causal suffix padding, which must be excluded from acceptance.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class RaggedGraphConfig:
    max_bs: int = 32
    token_multiple: int = 16
    max_padding_ratio: float = 0.125

    def __post_init__(self):
        if self.max_bs < 1:
            raise ValueError("Ragged graph max batch size must be positive.")
        if self.token_multiple < 1:
            raise ValueError("Ragged graph token multiple must be positive.")
        if not isfinite(self.max_padding_ratio) or not 0 <= self.max_padding_ratio <= 1:
            raise ValueError("Ragged graph padding ratio must be finite and in [0, 1].")

    @classmethod
    def from_server_args(cls, args):
        return cls(
            max_bs=args.speculative_ragged_cuda_graph_max_bs,
            token_multiple=args.speculative_ragged_cuda_graph_token_multiple,
            max_padding_ratio=args.speculative_ragged_cuda_graph_max_padding_ratio,
        )


def high_batch_fallback_width(raw: str, normal_k: int) -> Optional[int]:
    """Read the existing K:min_match environment setting for graph allocation."""
    if not raw.strip():
        return None
    try:
        width, match = (int(value) for value in raw.strip().split(":"))
    except ValueError as exc:
        raise ValueError(
            "SGLANG_DYNAMIC_K_HIGH_BATCH_FALLBACK must be K:min_match"
        ) from exc
    if width <= normal_k or match <= 0:
        raise ValueError(
            "High-batch fallback needs K > normal K and positive min_match"
        )
    return width


def dynamic_verify_widths(args, fallback_raw: str) -> Tuple[int, ...]:
    widths = {
        args.speculative_normal_draft_token_num,
        args.speculative_long_suffix_draft_token_num,
    }
    fallback = high_batch_fallback_width(
        fallback_raw, args.speculative_normal_draft_token_num
    )
    if fallback is not None:
        widths.add(fallback)
    return tuple(sorted(widths))


def choose_suffix_width(match_len, score, token_count, tiers, min_match_len):
    """Apply a common quality floor before any load-dependent tier selection."""
    if match_len < min_match_len:
        return None
    for width, tier_min_match in reversed(tiers):
        if (
            match_len >= tier_min_match
            and score >= width - 1
            and token_count >= width - 1
        ):
            return width
    return None


def bucket_key(
    widths: Sequence[int], normal_k: int, config: RaggedGraphConfig
) -> Optional[Tuple[int, int, int]]:
    """Return (exact batch size, execution token count, max query width).

    Homogeneous inputs use fixed-K graphs. More than two candidate widths
    are deliberately unsupported by this first bucket implementation.
    """
    bs = len(widths)
    if not 1 <= bs <= config.max_bs:
        return None
    max_k = max(widths)
    if set(widths) != {normal_k, max_k} or max_k <= normal_k:
        return None
    total = sum(widths)
    rounded = min(
        ((total + config.token_multiple - 1) // config.token_multiple)
        * config.token_multiple,
        bs * max_k,
    )
    if rounded - total > total * config.max_padding_ratio:
        return None
    return bs, rounded, max_k


def execution_widths(
    valid_widths: Sequence[int],
    total_tokens: int,
    max_k: int,
    capacities: Optional[Sequence[int]] = None,
) -> Optional[Tuple[int, ...]]:
    """Distribute synthetic tail tokens without crossing per-request capacity."""
    result = list(valid_widths)
    if not result or any(width < 1 or width > max_k for width in result):
        return None
    if capacities is not None and len(capacities) != len(result):
        raise ValueError("One capacity is required per request")
    limits = [
        max_k if capacities is None else min(max_k, capacities[i])
        for i in range(len(result))
    ]
    if any(width > limit for width, limit in zip(result, limits)):
        return None
    remaining = total_tokens - sum(result)
    if remaining < 0:
        return None
    for i, limit in enumerate(limits):
        extra = min(remaining, limit - result[i])
        result[i] += extra
        remaining -= extra
    return tuple(result) if remaining == 0 else None


def enumerate_bucket_shapes(
    normal_k: int,
    long_k: int,
    high_bs_threshold: int,
    fallback_k: Optional[int],
    config: RaggedGraphConfig,
    max_bs: Optional[int] = None,
):
    """Deduplicate all legal mixed shapes; no workload observation is needed."""
    shapes = {}
    limit = min(config.max_bs, max_bs if max_bs is not None else config.max_bs)
    for bs in range(2, limit + 1):
        width = long_k if bs < high_bs_threshold else fallback_k
        if width is None or width <= normal_k:
            continue
        for count in range(1, bs):
            valid = (width,) * count + (normal_k,) * (bs - count)
            key = bucket_key(valid, normal_k, config)
            # A full-width bucket already has a fixed-K graph. Do not capture
            # a second graph for the same model input shape.
            if key is not None and key[1] != bs * width:
                shapes.setdefault(key, execution_widths(valid, key[1], key[2]))
    return shapes


def build_bucket_inputs(draft_tokens, compact_map, valid_widths, exec_widths, seq_lens):
    """Materialize model input and maps, keeping synthetic tails out of drafts.

    Imported lazily so shape planning and configuration remain CPU-only and
    do not import the serving runtime (or require PyTorch).
    """
    import torch

    bs, max_k = compact_map.shape
    columns = torch.arange(max_k, device=draft_tokens.device)
    valid = compact_map >= 0
    execute = columns.unsqueeze(0) < exec_widths.unsqueeze(1)
    last = compact_map[
        torch.arange(bs, device=draft_tokens.device), valid_widths.long() - 1
    ]
    tokens = draft_tokens[last].unsqueeze(1).expand(bs, max_k).clone()
    tokens[valid] = draft_tokens[compact_map[valid]]
    cumulative = torch.nn.functional.pad(
        torch.cumsum(exec_widths, dim=0, dtype=torch.int32), (1, 0)
    )
    model_map = cumulative[:-1].long().unsqueeze(1) + columns
    # This map describes executed positions, including synthetic tails.
    # Acceptance validity comes from compact_map, which retains -1 in tails.
    model_map = model_map.masked_fill(~execute, -1)
    positions = seq_lens.long().unsqueeze(1) + columns
    return tokens[execute], positions[execute], cumulative, model_map
