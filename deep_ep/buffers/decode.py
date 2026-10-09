"""Backend-independent BF16 decode dispatch/combine.

Collective contract: all ranks use the same shapes, options and call order on
homogeneous devices. Routes must contain distinct, valid expert IDs per token.
Expert outputs are unweighted; combine applies the dispatch weights once.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist

from deep_ep import _C
from .ep import EPBuffer, EPHandle
from ..utils.event import EventOverlap


def _prefer_low_latency(ep: int, bs: int, hidden: int, topk: int, experts: int) -> bool:
    # Measured on A5 with balanced BF16 routes. Do not extrapolate to other
    # model shapes or EP256. BS is the actual local token count, not capacity.
    limit = 64 if ep == 8 else 32 if ep in (16, 32, 64, 128) else 0
    return (hidden, topk, experts) == (7168, 8, 128) and 0 < bs <= limit


@dataclass(frozen=True)
class DecodeHandle:
    """Routing state owned by a DecodeBuffer; pass unchanged to combine.

    Outputs are grouped by local expert. ``expert_recv_counts`` is an int64
    device tensor; only the leading sum(counts) rows are valid. CPU counts are
    populated only when dispatch requests ``do_cpu_sync=True``. ``backend`` is
    diagnostic information, never an input to the application workflow.
    """

    expert_recv_counts: torch.Tensor
    num_recv_tokens_per_expert_list: list
    num_expanded_tokens: int
    backend: str
    _owner: object = field(repr=False)
    _buffer: EPBuffer = field(repr=False)
    _handle: EPHandle = field(repr=False)
    _weights: Optional[torch.Tensor] = field(repr=False)


class DecodeBuffer:
    """Automatically select communication kernels for BF16 MoE decode.

    Unlike EPBuffer's general-purpose API, both implementations have the same
    weighted-combine contract here. No backend flag is required. Unsupported or
    unmeasured configurations use throughput kernels. No launch is retried on
    another backend: a communication failure requires restarting the job.

    Buffers are allocated collectively on first use of each backend and reused.
    The decode arena costs 8 GiB per rank plus workspace; after switching to a
    larger batch both arenas remain alive so outstanding handles remain valid.
    Destroy collectively, before destroying the process group.
    """

    def __init__(self, group: dist.ProcessGroup, num_max_tokens_per_rank: int,
                 hidden: int = 7168, num_topk: int = 8, num_experts: int = 128,
                 *, num_sms: int = 32,
                 use_huge1g_page: bool = True, num_cpu_timeout_secs: int = 300,
                 num_gpu_timeout_secs: int = 100):
        if (type(num_max_tokens_per_rank) is not int or num_max_tokens_per_rank <= 0
                or hidden <= 0 or hidden % 128 or not 1 <= num_topk <= 32
                or not group.size() <= num_experts <= 1024 or num_experts % group.size()
                or num_topk > num_experts
                or num_sms < 0 or num_cpu_timeout_secs <= 0 or num_gpu_timeout_secs <= 0):
            raise ValueError('Invalid decode capacity, model shape, cores or timeout')
        self.group = group
        self.num_ranks = group.size()
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.hidden, self.num_topk, self.num_experts = hidden, num_topk, num_experts
        self.num_sms = num_sms
        self.device = torch.device('npu', torch.npu.current_device())
        self._options = dict(use_huge1g_page=use_huge1g_page,
                             num_cpu_timeout_secs=num_cpu_timeout_secs,
                             num_gpu_timeout_secs=num_gpu_timeout_secs)
        self._buffers = {}
        self._choices = {}
        self._owner = object()
        self._destroyed = False
        self._failed = False
        self._last_event = None

    def _check_alive(self):
        if self._destroyed:
            raise RuntimeError('DecodeBuffer has been destroyed')
        if self._failed:
            raise RuntimeError('A decode launch failed; restart the distributed job')
        if torch.npu.current_device() != self.device.index:
            raise ValueError('Use the NPU on which this DecodeBuffer was constructed')

    def _check_tensor(self, tensor, shape, dtype, name):
        if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != tuple(shape)
                or tensor.dtype != dtype or tensor.device != self.device or not tensor.is_contiguous()):
            raise ValueError(f'{name} must be a contiguous {shape} {dtype} tensor on {self.device}')

    def _backend(self, bs):
        if bs not in self._choices:
            selected = 'throughput'
            if (_prefer_low_latency(self.num_ranks, bs, self.hidden, self.num_topk, self.num_experts)
                    and self.num_sms in (0, 32)
                    and _C.get_npu_arch() == 3510 and _C.get_num_vec_cores() >= 64):
                shape = (self.num_ranks, bs, self.hidden, self.num_topk, self.num_experts)
                # Both native plans must be admitted before allocating or launching.
                # A rejected plan is safe to fall back; an allocation/launch error is not.
                try:
                    if _C.low_latency_layout(*shape, 'bfloat16') is not None:
                        _C.low_latency_combine_tiling(*shape, 'bfloat16', bs)
                        selected = 'low_latency'
                except (ValueError, RuntimeError):
                    pass  # Host-only capability rejection (including insufficient UB).
            self._choices[bs] = selected
        return self._choices[bs]

    def _buffer(self, backend):
        if backend not in self._buffers:
            self._buffers[backend] = EPBuffer(
                self.group,
                num_max_tokens_per_rank=(min(256, self.num_max_tokens_per_rank)
                                         if backend == 'low_latency' else self.num_max_tokens_per_rank),
                hidden=self.hidden, num_topk=self.num_topk,
                low_latency_mode=backend == 'low_latency', explicitly_destroy=True, **self._options)
        return self._buffers[backend]

    def _begin(self):
        # Order both arenas if the caller changes the current stream.
        if self._last_event is not None:
            self._last_event.current_stream_wait()

    def dispatch(self, x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                 *, do_cpu_sync: bool = False):
        """Return ``(expert_x, handle, event)``; no backend-specific layout handling.

        Inputs: BF16 [B,H], int64 [B,K] valid unique expert IDs, FP32 [B,K]
        weights. All ranks must use identical B/H/K/E and options. The call does
        not inspect device route values or add a host agreement collective.
        Keep route tensors unchanged while the handle is in use. All work is
        enqueued on the current stream. First use includes collective allocation
        and JIT; warm up each required batch/backend before timing. Graph capture
        is not currently supported by this adapter.
        """
        self._check_alive()
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            raise ValueError('x must have shape [B,H]')
        bs = x.shape[0]
        if not 0 < bs <= self.num_max_tokens_per_rank:
            raise ValueError('Require 0 < B <= num_max_tokens_per_rank')
        self._check_tensor(x, (bs, self.hidden), torch.bfloat16, 'x')
        self._check_tensor(topk_idx, (bs, self.num_topk), torch.int64, 'topk_idx')
        self._check_tensor(topk_weights, (bs, self.num_topk), torch.float32, 'topk_weights')
        backend = self._backend(bs)
        self._begin()
        try:
            buffer = self._buffer(backend)
            recv, _, weights, handle, event = buffer.dispatch(
                x, topk_idx=topk_idx, topk_weights=topk_weights, num_experts=self.num_experts,
                # The LL direct/dedup split must follow this batch, not a larger
                # constructor capacity or an earlier dispatch.
                num_max_tokens_per_rank=bs if backend == 'low_latency' else self.num_max_tokens_per_rank,
                expert_alignment=1, num_sms=self.num_sms,
                do_expand=True, do_cpu_sync=do_cpu_sync,
                async_with_compute_stream=True)
            counts = (handle.expert_recv_counts if backend == 'low_latency'
                      else handle.num_unaligned_recv_tokens_per_expert.to(torch.int64))
            # The count conversion must be included in the public event.
            event = EventOverlap(EPBuffer.capture(), (recv, counts, weights, handle, event))
            result = DecodeHandle(counts, handle.num_recv_tokens_per_expert_list,
                                  handle.num_expanded_tokens, backend, self._owner, buffer, handle, weights)
            self._last_event = event.event
        except Exception:
            self._failed = True
            raise
        return recv, result, event

    def combine(self, x: torch.Tensor, handle: DecodeHandle):
        """Return ``(combined_x, event)`` and apply saved routing weights once.

        ``x`` is an unweighted BF16 expert output with the dispatch output's
        shape/order. A handle continues to use its originating backend even
        after other batch sizes have selected another implementation. Repeated
        combine is supported and does not modify x or the saved weights.
        """
        self._check_alive()
        if not isinstance(handle, DecodeHandle) or handle._owner is not self._owner:
            raise ValueError('combine requires a dispatch handle from this DecodeBuffer')
        self._check_tensor(x, (handle.num_expanded_tokens, self.hidden), torch.bfloat16, 'expert output')
        self._begin()
        try:
            stream = torch.npu.current_stream()
            # PyTorch's allocator needs the consumer stream as well as the
            # creation stream when callers discard a handle after submission.
            x.record_stream(stream)
            if handle._weights is not None:
                handle._weights.record_stream(stream)
            for value in vars(handle._handle).values():
                if isinstance(value, torch.Tensor):
                    value.record_stream(stream)
            weighted = (x.float() * handle._weights.unsqueeze(1)).to(x.dtype) if handle._weights is not None else x
            output, _, event = handle._buffer.combine(
                weighted, handle._handle, num_sms=self.num_sms, async_with_compute_stream=True)
            # Keep the temporary weighted rows and originating arena/metadata
            # alive until the enqueued communication has completed.
            weighted.record_stream(stream)
            event.extra_tensors = (x, weighted, handle, event.extra_tensors)
            self._last_event = event.event
        except Exception:
            self._failed = True
            raise
        return output, event

    def destroy(self):
        """Collectively destroy every materialized arena; safe to call twice."""
        if self._destroyed:
            return
        self._begin()
        for backend in sorted(self._buffers):
            self._buffers[backend].destroy()
        self._buffers.clear()
        self._last_event = None
        self._destroyed = True
