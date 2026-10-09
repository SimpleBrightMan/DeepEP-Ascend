"""CPU-only admission, backend lifetime and failure checks for DecodeBuffer."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_low_latency_host import Device, Event, Tensor, fake_modules, load


class DecodeContracts(unittest.TestCase):
    def setUp(self):
        import sys
        modules, self.torch, self.native = fake_modules()
        self.patcher = patch.dict(sys.modules, modules)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        load('deep_ep.buffers.base', 'deep_ep/buffers/base.py')
        load('deep_ep.utils.event', 'deep_ep/utils/event.py')
        load('deep_ep.buffers.ep', 'deep_ep/buffers/ep.py')
        self.api = load('deep_ep.buffers.decode', 'deep_ep/buffers/decode.py')
        self.native.get_npu_arch = lambda: 3510
        self.native.get_num_vec_cores = lambda: 64
        self.group = SimpleNamespace(size=lambda: 8)
        self.buffer = self.api.DecodeBuffer(self.group, 4096)

    def test_measured_thresholds_and_unmeasured_models(self):
        prefer = self.api._prefer_low_latency
        for ep, boundary in [(8, 64), (16, 32), (32, 32), (64, 32), (128, 32)]:
            for bs in [0, 1, 2, 8, 16, 32, 33, 64, 65, 128, 256, 4096]:
                with self.subTest(ep=ep, bs=bs):
                    self.assertEqual(prefer(ep, bs, 7168, 8, 128), 0 < bs <= boundary)
        for shape in [(256, 2, 7168, 8, 128), (4, 2, 7168, 8, 128),
                      (8, 2, 4096, 8, 128), (8, 2, 7168, 6, 128), (8, 2, 7168, 8, 256)]:
            self.assertFalse(prefer(*shape))

    def test_admission_uses_actual_bs_and_checks_complete_pair(self):
        with patch.object(self.native, 'low_latency_combine_tiling', return_value=b'plan') as plan:
            self.assertEqual(self.buffer._backend(16), 'low_latency')
            self.assertEqual(self.buffer._backend(128), 'throughput')
            self.assertEqual(self.buffer._backend(16), 'low_latency')
        self.assertEqual(plan.call_count, 1)
        self.assertEqual(plan.call_args.args, (8, 16, 7168, 8, 128, 'bfloat16', 16))

    def test_capability_rejection_and_unmeasured_cores_fall_back(self):
        with patch.object(self.native, 'low_latency_layout', return_value=None):
            self.assertEqual(self.buffer._backend(2), 'throughput')
        with patch.object(self.native, 'low_latency_combine_tiling', side_effect=ValueError('UB')):
            self.assertEqual(self.buffer._backend(8), 'throughput')
        with patch.object(self.native, 'get_npu_arch', return_value=3500):
            self.assertEqual(self.buffer._backend(16), 'throughput')
        self.buffer.num_sms = 16
        self.assertEqual(self.buffer._backend(32), 'throughput')
        self.assertEqual(self.buffer._buffers, {})

    def test_lazy_arenas_are_reused_and_destroyed(self):
        arenas = []
        def create(*args, **kwargs):
            arena = SimpleNamespace(options=kwargs, destroy=lambda: None)
            arenas.append(arena)
            return arena
        with patch.object(self.api, 'EPBuffer', side_effect=create):
            ll = self.buffer._buffer('low_latency')
            ht = self.buffer._buffer('throughput')
            self.assertIs(self.buffer._buffer('low_latency'), ll)
        self.assertEqual(len(arenas), 2)
        self.assertEqual(ll.options['num_max_tokens_per_rank'], 256)
        self.assertEqual(ht.options['num_max_tokens_per_rank'], 4096)
        self.buffer.destroy()
        self.buffer.destroy()
        self.assertEqual(self.buffer._buffers, {})
        with self.assertRaisesRegex(RuntimeError, 'destroyed'):
            self.buffer._check_alive()

    def test_combine_uses_originating_handle_after_backend_switch(self):
        calls = []
        original = SimpleNamespace()
        arena = SimpleNamespace(combine=lambda x, h, **kw: (
            calls.append(h) or x, None, self.api.EventOverlap(Event())))
        handle = self.api.DecodeHandle(Tensor((16,), self.torch.int64), [], 4,
                                       'low_latency', self.buffer._owner, arena, original, None)
        x = Tensor((4, 7168), device=Device('npu', 0))
        self.assertEqual(self.buffer._backend(128), 'throughput')
        for _ in range(2):
            out, _ = self.buffer.combine(x, handle)
            self.assertIs(out, x)
        self.assertEqual(calls, [original, original])
        other = self.api.DecodeBuffer(self.group, 128)
        with self.assertRaisesRegex(ValueError, 'from this'):
            other.combine(x, handle)

    def test_failure_is_never_retried_with_throughput(self):
        arena = SimpleNamespace(dispatch=lambda *a, **k: (_ for _ in ()).throw(RuntimeError('launch failed')))
        self.buffer._buffers['low_latency'] = arena
        args = [Tensor((2, 7168), device=Device('npu', 0)),
                Tensor((2, 8), self.torch.int64, Device('npu', 0)),
                Tensor((2, 8), self.torch.float32, Device('npu', 0))]
        with self.assertRaisesRegex(RuntimeError, 'launch failed'):
            self.buffer.dispatch(*args)
        self.assertNotIn('throughput', self.buffer._buffers)
        with self.assertRaisesRegex(RuntimeError, 'restart'):
            self.buffer.dispatch(*args)


if __name__ == '__main__':
    unittest.main()
