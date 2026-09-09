"""CPU planning/mapping tests plus an optional CUDA FA3 replay check.

Run directly to avoid importing the GPU serving runtime on a CPU machine:
    python test/srt/test_ragged_cuda_graph.py
"""

import importlib.util
import itertools
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "ragged_graph_planner",
    ROOT / "python/sglang/srt/speculative/ragged_cuda_graph.py",
)
planner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)

try:
    import torch
except ImportError:
    torch = None


class RaggedPlanningTests(unittest.TestCase):
    def test_common_match_floor_for_both_load_branches(self):
        for width in (8, 16):
            for match in (8, 22, 23, 24):
                result = planner.choose_suffix_width(
                    match, 30.0, 30, [(width, 8)], min_match_len=23
                )
                self.assertEqual(result, width if match >= 23 else None)
        self.assertIsNone(planner.choose_suffix_width(23, 14.9, 15, [(16, 23)], 23))
        self.assertIsNone(planner.choose_suffix_width(23, 15, 14, [(16, 23)], 23))
        self.assertEqual(planner.choose_suffix_width(23, 15, 15, [(16, 23)], 23), 16)

    def test_fixed_graphs_include_high_batch_width(self):
        args = SimpleNamespace(
            speculative_normal_draft_token_num=4,
            speculative_long_suffix_draft_token_num=16,
        )
        self.assertEqual(planner.dynamic_verify_widths(args, "8:23"), (4, 8, 16))
        self.assertEqual(planner.dynamic_verify_widths(args, ""), (4, 16))
        for raw in ("8", "a:23", "8:0", "4:23", "8:23:4"):
            with self.assertRaises(ValueError):
                planner.dynamic_verify_widths(args, raw)

    def test_examples_and_padding_boundary(self):
        config = planner.RaggedGraphConfig()
        self.assertEqual(
            planner.bucket_key([16] * 3 + [4] * 7, 4, config), (10, 80, 16)
        )
        self.assertIsNone(planner.bucket_key([16] + [4] * 9, 4, config))
        # Small active batches follow the same token buckets, without an exact path.
        self.assertIsNone(planner.bucket_key([16, 4, 4, 4], 4, config))
        self.assertEqual(planner.bucket_key([16, 4, 4, 4, 4], 4, config), (5, 32, 16))
        self.assertIsNone(planner.bucket_key([4] * 10, 4, config))
        self.assertIsNone(planner.bucket_key([8] * 24, 4, config))
        self.assertIsNone(planner.bucket_key([4, 8, 16], 4, config))
        self.assertIsNone(planner.bucket_key([8] + [4] * 32, 4, config))
        # 32 real tokens rounded to 36 has exactly 12.5% overhead.
        boundary = planner.RaggedGraphConfig(token_multiple=9)
        self.assertEqual(planner.bucket_key([16, 4, 4, 4, 4], 4, boundary), (5, 36, 16))

    def test_context_capacity_can_move_padding_or_fall_back(self):
        self.assertEqual(planner.execution_widths([16, 4, 4], 28, 16), (16, 8, 4))
        self.assertEqual(
            planner.execution_widths([16, 4, 4], 28, 16, [16, 4, 8]), (16, 4, 8)
        )
        self.assertIsNone(planner.execution_widths([16, 4, 4], 28, 16, [16, 4, 4]))
        self.assertIsNone(planner.execution_widths([16, 4], 19, 16))
        self.assertIsNone(planner.execution_widths([16, 4], 33, 16))

    def test_enumeration_covers_runtime_keys_in_any_order(self):
        config = planner.RaggedGraphConfig()
        shapes = planner.enumerate_bucket_shapes(4, 16, 24, 8, config)
        self.assertEqual(len(shapes), 233)
        rng = random.Random(7)
        for bs in range(2, 33):
            long_k = 16 if bs < 24 else 8
            for count in range(1, bs):
                widths = [long_k] * count + [4] * (bs - count)
                rng.shuffle(widths)
                key = planner.bucket_key(widths, 4, config)
                if key is None:
                    continue
                if key[1] != bs * long_k:
                    self.assertIn(key, shapes)
                physical = planner.execution_widths(widths, key[1], key[2])
                self.assertEqual(sum(physical), key[1])
                self.assertEqual(max(physical), key[2])
                self.assertTrue(all(a <= b <= long_k for a, b in zip(widths, physical)))
                self.assertLessEqual(sum(physical) - sum(widths), sum(widths) / 8)
                if key in shapes:
                    self.assertEqual(sum(shapes[key]), key[1])
        self.assertTrue(all(key[2] == 8 for key in shapes if key[0] >= 24))
        self.assertTrue(
            all(
                key[0] < 24
                for key in planner.enumerate_bucket_shapes(4, 16, 24, None, config)
            )
        )
        self.assertTrue(
            all(
                key[0] <= 4
                for key in planner.enumerate_bucket_shapes(4, 16, 24, 8, config, 4)
            )
        )

    def test_invalid_config(self):
        for kwargs in (
            {"max_bs": 0},
            {"token_multiple": 0},
            {"max_padding_ratio": -1},
            {"max_padding_ratio": float("nan")},
            {"max_padding_ratio": float("inf")},
        ):
            with self.assertRaises(ValueError):
                planner.RaggedGraphConfig(**kwargs)


@unittest.skipIf(torch is None, "PyTorch is needed for tensor mapping tests")
class RaggedMappingTests(unittest.TestCase):
    def make_inputs(self, widths, total, device="cpu"):
        max_k = max(widths)
        physical = planner.execution_widths(widths, total, max_k)
        compact = torch.full((len(widths), max_k), -1, dtype=torch.long, device=device)
        offset = 0
        for i, width in enumerate(widths):
            compact[i, :width] = torch.arange(offset, offset + width, device=device)
            offset += width
        tokens = torch.arange(offset, dtype=torch.long, device=device) + 100
        valid = torch.tensor(widths, dtype=torch.int32, device=device)
        execute = torch.tensor(physical, dtype=torch.int32, device=device)
        seq_lens = torch.arange(len(widths), device=device, dtype=torch.int32) + 20
        result = planner.build_bucket_inputs(tokens, compact, valid, execute, seq_lens)
        return compact, tokens, physical, result

    def test_all_small_permutations_preserve_real_tokens_and_positions(self):
        for widths in set(itertools.permutations([16, 4, 4])):
            compact, tokens, physical, (ids, positions, cu, mapping) = self.make_inputs(
                widths, 28
            )
            self.assertEqual(ids.numel(), 28)
            self.assertEqual(cu.tolist(), [0, physical[0], sum(physical[:2]), 28])
            for i, width in enumerate(widths):
                indices = mapping[i, :width]
                torch.testing.assert_close(ids[indices], tokens[compact[i, :width]])
                self.assertEqual(
                    positions[indices].tolist(), list(range(20 + i, 20 + i + width))
                )
                tail = mapping[i, width : physical[i]]
                self.assertTrue(torch.all(ids[tail] == tokens[compact[i, width - 1]]))

    def test_rejected_and_synthetic_kv_slots_are_all_reclaimed(self):
        widths = [4, 16, 4]
        compact, _, physical, (_, _, _, mapping) = self.make_inputs(widths, 28)
        # Independent oracle: keep root plus a valid accepted prefix. Include
        # immediate rejection, partial acceptance, and complete acceptance.
        for kept_lengths in ([1, 1, 1], [4, 7, 2], [4, 16, 4]):
            accepted = torch.cat([mapping[i, :n] for i, n in enumerate(kept_lengths)])
            evict = torch.ones(28, dtype=torch.bool)
            evict[accepted] = False
            expected = set()
            offset = 0
            for width, kept in zip(physical, kept_lengths):
                expected.update(range(offset + kept, offset + width))
                offset += width
            self.assertEqual(set(evict.nonzero().flatten().tolist()), expected)
            synthetic = mapping[(compact < 0) & (mapping >= 0)]
            self.assertTrue(torch.all(evict[synthetic]))
            self.assertEqual(len(accepted), sum(kept_lengths))

    def test_causal_attention_real_prefix_is_unchanged(self):
        torch.manual_seed(42)
        widths = [4, 16, 4]
        compact, _, physical, (_, _, _, mapping) = self.make_inputs(widths, 28)
        query = torch.randn(sum(widths), 8)
        keys, values = torch.randn(sum(widths), 8), torch.randn(sum(widths), 8)
        for i, (valid, execute) in enumerate(zip(widths, physical)):
            indices = compact[i, :valid]
            q, k, v = query[indices], keys[indices], values[indices]
            mask = torch.ones(valid, valid, dtype=torch.bool).triu(1)
            expected = (q @ k.T / 8**0.5).masked_fill(mask, -torch.inf).softmax(-1) @ v
            qx = torch.cat([q, torch.randn(execute - valid, 8)])
            kx = torch.cat([k, torch.randn(execute - valid, 8)])
            vx = torch.cat([v, torch.randn(execute - valid, 8)])
            maskx = torch.ones(execute, execute, dtype=torch.bool).triu(1)
            actual = (qx @ kx.T / 8**0.5).masked_fill(maskx, -torch.inf).softmax(
                -1
            ) @ vx
            torch.testing.assert_close(actual[:valid], expected)

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(), "CUDA FA3 runtime required"
    )
    def test_fa3_graph_replay_with_different_row_boundaries(self):
        from sgl_kernel.flash_attn import flash_attn_with_kvcache

        torch.manual_seed(5)
        # One graph shape: B=3, T=28, max Q=16. Real token count is 24.
        query = torch.randn(28, 2, 64, device="cuda", dtype=torch.float16)
        key_cache = torch.randn(96, 1, 2, 64, device="cuda", dtype=torch.float16)
        value_cache = torch.randn_like(key_cache)
        table = torch.arange(96, device="cuda", dtype=torch.int32).view(3, 32)
        cu_q = torch.tensor([0, 16, 24, 28], device="cuda", dtype=torch.int32)
        cache_lens = torch.tensor([32, 24, 20], device="cuda", dtype=torch.int32)
        cu_k = torch.nn.functional.pad(cache_lens.cumsum(0, dtype=torch.int32), (1, 0))

        def forward(q=query, boundaries=cu_q, lengths=cache_lens, kv_boundaries=cu_k):
            return flash_attn_with_kvcache(
                q=q,
                k_cache=key_cache,
                v_cache=value_cache,
                page_table=table,
                cache_seqlens=lengths,
                cu_seqlens_q=boundaries,
                cu_seqlens_k_new=kv_boundaries,
                max_seqlen_q=16,
                causal=True,
                num_splits=0,
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                forward()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = forward()

        for widths in ([16, 4, 4], [4, 16, 4], [4, 4, 16]):
            compact, _, physical, (_, _, boundaries, mapping) = self.make_inputs(
                widths, 28, "cuda"
            )
            query.copy_(torch.randn_like(query))
            cu_q.copy_(boundaries)
            cache_lens.copy_(
                torch.tensor(physical, device="cuda", dtype=torch.int32) + 16
            )
            cu_k[1:].copy_(cache_lens.cumsum(0, dtype=torch.int32))
            graph.replay()
            expected = forward()
            torch.testing.assert_close(graph_output, expected, rtol=2e-3, atol=2e-3)
            # Independent compact eager call excludes every synthetic tail.
            real_indices = mapping[compact >= 0]
            real_widths = torch.tensor(widths, device="cuda", dtype=torch.int32)
            real_cu_q = torch.nn.functional.pad(
                real_widths.cumsum(0, dtype=torch.int32), (1, 0)
            )
            real_lens = real_widths + 16
            real_cu_k = torch.nn.functional.pad(
                real_lens.cumsum(0, dtype=torch.int32), (1, 0)
            )
            expected_real = forward(
                query[real_indices], real_cu_q, real_lens, real_cu_k
            )
            torch.testing.assert_close(
                graph_output[real_indices], expected_real, rtol=2e-3, atol=2e-3
            )


if __name__ == "__main__":
    unittest.main()
