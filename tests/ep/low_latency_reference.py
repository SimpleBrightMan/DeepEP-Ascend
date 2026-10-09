# Copyright (c) 2026, Lu Lu

"""Independent CPU reference helpers from Ascend/DeepEP, synchronized at 8ffafeb."""


import random


def make_routes(world, b, k, e, seed):
    """Independent CPU routes: no repeated experts within any token."""
    rng = random.Random(seed)
    result = []
    for _ in range(world):
        rows = []
        for _ in range(b):
            chosen = rng.sample(range(e), k)
            rows.append(chosen)
        result.append(rows)
    return result


def expert_entries(routes, experts, world, rank):
    local = experts // world
    result = [[] for _ in range(local)]
    for source, rows in enumerate(routes):
        for token, selected in enumerate(rows):
            for expert in selected:
                if rank * local <= expert < (rank + 1) * local:
                    result[expert - rank * local].append((source, token))
    return result


def payload(sources, tokens, hidden, dtype):
    import torch

    sources, tokens = torch.as_tensor(sources), torch.as_tensor(tokens)
    cols = torch.arange(hidden)
    values = ((sources[:, None] * 3 + tokens[:, None] * 5 + cols) % 17 - 8).to(dtype)
    # Four small exact integers identify every row, even with FP8 input.
    values[:, 0], values[:, 1] = sources // 16, sources % 16
    values[:, 2], values[:, 3] = tokens // 16, tokens % 16
    return values


def fp8_scales(sources, tokens, hidden, scale_dtype):
    import torch

    groups = (hidden + 127) // 128
    if scale_dtype == "int32":
        return torch.full((len(sources), groups), 0x7F7F7F7F, dtype=torch.int32)
    # Finite, positive non-unit values with distinct low mantissa bits per row/group.
    bits = (
        0x3F000001
        + torch.as_tensor(sources)[:, None] * 65536
        + torch.as_tensor(tokens)[:, None] * 128
        + torch.arange(groups)
    )
    return bits.to(torch.int32).contiguous().view(torch.float32)


def expected_combine(routes, rank, hidden, dtype, fp32_scales=False):
    import torch

    b, k = len(routes[rank]), len(routes[rank][0])
    x = payload([rank] * b, list(range(b)), hidden, dtype).float()
    if fp32_scales:
        x *= fp8_scales([rank] * b, list(range(b)), hidden, "float32").repeat_interleave(128, dim=1)[:, :hidden]
    expected = torch.zeros_like(x)
    for token in range(b):
        for slot in range(k):
            expert = routes[rank][token][slot]
            if expert >= 0:
                expert_value = (x[token] * ((expert % 4 + 1) / 4) + (expert % 3 - 1) / 8).to(dtype).float()
                expected[token] += expert_value * ((slot + 1) / 32)
    return expected.to(dtype)


def prepare_experts(recv, counts, rank, experts, world, expert_dtype):
    import torch

    values = recv[0] if isinstance(recv, tuple) else recv
    output = torch.zeros(values.shape, dtype=expert_dtype, device=values.device)
    start = 0
    for local, count in enumerate(counts):
        expert = rank * (experts // world) + local
        block = values[start : start + count].float()
        if isinstance(recv, tuple) and recv[1].dtype == torch.float32:
            block = block * recv[1][start : start + count].repeat_interleave(128, dim=1)[:, : values.shape[1]]
        output[start : start + count] = (block * ((expert % 4 + 1) / 4) + (expert % 3 - 1) / 8).to(expert_dtype)
        start += count
    return output
