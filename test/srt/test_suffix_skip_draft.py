"""Exercise worker dispatch without importing CUDA/Triton serving dependencies.

The actual worker methods are extracted with AST; model forwards are test doubles.
Run: python test/srt/test_suffix_skip_draft.py
"""

import ast
import copy
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[2]
source = ast.parse((ROOT / "python/sglang/srt/speculative/eagle_worker.py").read_text())
worker = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "EAGLEWorker")


def method(name, **globals_):
    node = next(n for n in worker.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node = copy.deepcopy(node)
    for arg in node.args.args:
        arg.annotation = None
    node.returns = None
    namespace = dict(copy=copy, **globals_)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "worker-under-test", "exec"), namespace)
    return namespace[name]


class DispatchTests(unittest.TestCase):
    def make_case(self, selected):
        batch = NS(
            forward_mode=NS(is_idle=lambda: False),
            batch_size=lambda: 3,
            spec_info=NS(future_indices=None),
            sampling_info=NS(penalizer_orchestrator=NS(is_required=False)),
        )
        w = NS(
            server_args=NS(speculative_suffix_skip_draft=True, enable_dp_attention=False),
            model_config=NS(is_encoder_decoder=False),
            speculative_algorithm=NS(is_standalone=lambda: True),
            _long_suffix_draft_token_num=16,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            _get_suffix_proposals=Mock(return_value=["a", "b", "c"]),
            _select_suffix_draft_token_nums=Mock(return_value=selected),
            _can_use_ragged_dynamic_k=Mock(return_value=bool(selected)),
            _draft_suffix_first_candidates=Mock(return_value=("p", "s", "t")),
            _draft_model_candidates=Mock(return_value=("p", "s", "t")),
            _apply_suffix_overrides=Mock(),
            _build_ragged_verify_input=Mock(return_value="ragged"),
            _build_standalone_verify_input=Mock(return_value="normal"),
        )
        return w, batch

    def test_mixed_and_all_hit_use_same_target_builder(self):
        for selected in ({1: 16}, {0: 16, 1: 8, 2: 16}):
            w, batch = self.make_case(selected)
            self.assertEqual(method("draft")(w, batch), "ragged")
            w._draft_model_candidates.assert_not_called()
            w._draft_suffix_first_candidates.assert_called_once_with(batch, selected)
            self.assertEqual(w._build_ragged_verify_input.call_args.args[-1], selected)
            w._get_suffix_proposals.assert_called_once()

    def test_no_hit_retains_regular_draft(self):
        w, batch = self.make_case({})
        self.assertEqual(method("draft")(w, batch), "normal")
        w._draft_model_candidates.assert_called_once_with(batch)
        w._draft_suffix_first_candidates.assert_not_called()

    def test_disabled_or_unsupported_paths_fall_back(self):
        for field, value in (
            ("speculative_suffix_skip_draft", False),
            ("enable_dp_attention", True),
            ("dp_size", 2),
            ("pp_size", 2),
        ):
            w, batch = self.make_case({1: 16})
            setattr(w.server_args, field, value)
            method("draft")(w, batch)
            w._draft_model_candidates.assert_called_once()
            w._draft_suffix_first_candidates.assert_not_called()
        for condition in ("penalties", "future", "encoder", "algorithm"):
            w, batch = self.make_case({1: 16})
            if condition == "penalties":
                batch.sampling_info.penalizer_orchestrator.is_required = True
            elif condition == "future":
                batch.spec_info.future_indices = object()
            elif condition == "encoder":
                w.model_config.is_encoder_decoder = True
            else:
                w.speculative_algorithm.is_standalone = lambda: False
            method("draft")(w, batch)
            w._draft_suffix_first_candidates.assert_not_called()


class Tensor:
    """Small indexing double; model numerics are deliberately outside this test."""
    def __init__(self, data):
        self.data = data
        self.dtype = "int64"

    def expand(self, rows, _):
        return Tensor([self.data[:] for _ in range(rows)])

    def __sub__(self, value):
        return Tensor([[x - value for x in row] for row in self.data])

    def clone(self):
        return Tensor(copy.deepcopy(self.data))

    def __setitem__(self, indices, values):
        for i, value in zip(indices.data, values.data):
            self.data[i] = value[:]


TORCH = NS(
    long="int64",
    zeros=lambda shape, **_: Tensor([[0] * shape[1] for _ in range(shape[0])]),
    arange=lambda size, **_: Tensor(list(range(size))),
    tensor=lambda data, **_: Tensor(data),
)


class DraftState:
    def __init__(self):
        self.topk_index = Tensor([[11], [21], [31]])
        self.verified_id = [10, 20, 30]

    def filter_batch(self, indices, has_been_filtered):
        assert not has_been_filtered
        self.topk_index = Tensor([self.topk_index.data[i] for i in indices.data])
        self.verified_id = [self.verified_id[i] for i in indices.data]


class CompactionTests(unittest.TestCase):
    def test_fallback_rows_scatter_in_original_order_without_mutating_full_state(self):
        batch = NS(batch_size=lambda: 3, seq_lens=NS(device="cpu"), spec_info=DraftState())
        seen = []

        def draft(sub):
            seen.append(sub.spec_info.verified_id)
            self.assertEqual(sub.input_ids, [10, 30])
            return None, None, Tensor([[11, 12, 13], [31, 32, 33]])

        w = NS(speculative_num_draft_tokens=4,
               _make_sub_batch=lambda batch, indices: copy.copy(batch),
               _draft_model_candidates=draft)
        _, _, tokens = method("_draft_suffix_first_candidates", torch=TORCH)(w, batch, {1: 16})
        self.assertEqual(tokens.data, [[11, 12, 13], [0, 0, 0], [31, 32, 33]])
        self.assertEqual(seen, [[10, 30]])
        self.assertEqual(batch.spec_info.verified_id, [10, 20, 30])
        self.assertEqual(batch.spec_info.topk_index.data, [[11], [21], [31]])
        self.assertEqual(w._suffix_draft_skipped_request_count, 1)

    def test_all_hit_never_constructs_or_executes_draft_sub_batch(self):
        batch = NS(batch_size=lambda: 3, seq_lens=NS(device="cpu"), spec_info=DraftState())
        w = NS(speculative_num_draft_tokens=4, _make_sub_batch=Mock(), _draft_model_candidates=Mock())
        method("_draft_suffix_first_candidates", torch=TORCH)(w, batch, {0: 16, 1: 16, 2: 8})
        w._make_sub_batch.assert_not_called()
        w._draft_model_candidates.assert_not_called()
        self.assertEqual(w._suffix_draft_skipped_request_count, 3)


class LifecycleTests(unittest.TestCase):
    def test_target_rejection_still_updates_draft_and_finished_rows_do_not_extend(self):
        for remaining in (0, 2):
            events = []
            batch = NS(forward_mode=NS(is_extend=lambda: False), is_extend_in_batch=False,
                       batch_size=lambda: 3, spec_info=None)
            w = NS(_suffix_proposer=None, server_args=NS(enable_dp_attention=False),
                   draft_tp_context=lambda _: nullcontext(),
                   draft_model_runner=NS(tp_group=None))

            def draft(batch):
                events.append("draft-skip")
                w._suffix_draft_skipped_request_count = 3
                w._suffix_draft_skipped_batch_count = 1
                return "suffix-candidates"

            def verify(batch, candidates):
                self.assertEqual(candidates, "suffix-candidates")
                events.append("target-verify")
                # Target accepted variable lengths, not all suffix candidates.
                batch.spec_info = NS(verified_id=NS(shape=(remaining,)))
                return None, NS(verified_id=[10, 20, 21, 30], accept_length_per_req_cpu=[0, 1, 0]), None, False

            w.draft = draft
            w.verify = verify
            w._get_num_verify_tokens = lambda *_: 48
            w.forward_draft_extend_after_decode = lambda _: events.append("draft-extend")
            result = method("forward_batch_generation", GenerationBatchResult=NS,
                            speculative_moe_backend_context=nullcontext)(w, batch)
            self.assertEqual(events, ["draft-skip", "target-verify"] + (["draft-extend"] if remaining else []))
            self.assertEqual(result.num_accepted_tokens, 1)
            self.assertEqual(result.suffix_draft_skipped_request_count, 3)


if __name__ == "__main__":
    unittest.main()
