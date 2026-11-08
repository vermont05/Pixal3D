# Pure-PyTorch stand-in for NATTEN's na2d (neighborhood attention), for GPUs without NATTEN kernels (AMD/ROCm).
# Same semantics as natten.na2d for stride 1, non-causal: each query attends to a kernel_size window of keys on its
# dilation sub-grid; near the borders the window shifts inward (it is never padded). Layout [B, H, W, heads, dim];
# the value head dim may differ from the query/key head dim. Computed in row chunks to bound memory.
import torch

__version__ = "0.0-torch-shim"
HAS_LIBNATTEN = False


def _axis_neighbors(L: int, k: int, d: int, device) -> torch.Tensor:
    """[L, k] neighbor indices along one axis (NATTEN window rules with dilation)."""
    i = torch.arange(L, device=device)
    g = i % d                                   # dilation group
    j = i // d                                  # index inside the group
    Lg = (L - g + d - 1) // d                   # group length
    if int(Lg.min()) < k:
        raise ValueError(f"na2d shim: axis length {L} too small for kernel {k} with dilation {d}")
    start = (j - k // 2).clamp(min=0)
    start = torch.minimum(start, Lg - k)
    return g[:, None] + d * (start[:, None] + torch.arange(k, device=device)[None])


def na2d(query, key, value, kernel_size, dilation=1, stride=1, is_causal=False, scale=None, backend=None, **kwargs):
    if stride not in (1, (1, 1)) or is_causal not in (False, (False, False)):
        raise NotImplementedError("na2d shim supports stride=1, non-causal only")
    ks = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    dl = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
    B, H, W, N, D = query.shape
    Dv = value.shape[-1]
    scale = D ** -0.5 if scale is None else scale
    ry = _axis_neighbors(H, ks[0], dl[0], query.device)   # [H, ky]
    rx = _axis_neighbors(W, ks[1], dl[1], query.device)   # [W, kx]
    K = ks[0] * ks[1]
    kf = key.reshape(B, H * W, N, D)
    vf = value.reshape(B, H * W, N, Dv)
    out = torch.empty(B, H, W, N, Dv, dtype=value.dtype, device=value.device)
    per_row = B * W * K * N * (D + Dv) * kf.element_size()
    rows = max(1, min(H, int(512 * 2**20 // max(per_row, 1))))
    for y0 in range(0, H, rows):
        y1 = min(H, y0 + rows)
        idx = (ry[y0:y1, None, :, None] * W + rx[None, :, None, :]).reshape(-1)   # [(r*W*ky*kx)]
        kn = kf.index_select(1, idx).view(B, y1 - y0, W, K, N, D)
        vn = vf.index_select(1, idx).view(B, y1 - y0, W, K, N, Dv)
        q = query[:, y0:y1]                                                   # [B, r, W, N, D]
        att = torch.einsum("brwnd,brwknd->brwnk", q.float(), kn.float()) * scale
        att = att.softmax(dim=-1)
        out[:, y0:y1] = torch.einsum("brwnk,brwknd->brwnd", att, vn.float()).to(out.dtype)
    return out


def na2d_lowres(query, key_lr, value_lr, kernel_size, scale=None, out_dtype=torch.bfloat16):
    """na2d(query, upsample(key_lr), upsample(value_lr), dilation=H//h) without materialising the nearest-exact upsampled
    key/value maps: on the dilation sub-grid of an upsampled map, positions g + d*m hold low-res pixel m. Returns [B, N*Dv, H, W]."""
    ks = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    B, H, W, N, D = query.shape
    h, w = key_lr.shape[1:3]
    Dv = value_lr.shape[-1]
    if H % h or W % w:
        raise ValueError("na2d_lowres needs an integer upsampling factor")
    scale = D ** -0.5 if scale is None else scale
    dev = query.device
    def win(L, l, k):
        s = (torch.arange(L, device=dev) // (L // l) - k // 2).clamp(min=0).clamp(max=l - k)
        return s[:, None] + torch.arange(k, device=dev)[None]
    ry, rx = win(H, h, ks[0]), win(W, w, ks[1])
    K = ks[0] * ks[1]
    kf = key_lr.reshape(B, h * w, N, D).float()
    vf = value_lr.reshape(B, h * w, N, Dv).float()
    out = torch.empty(B, N * Dv, H, W, dtype=out_dtype, device=dev)
    per_row = B * W * K * N * (D + Dv) * 4
    rows = max(1, min(H, int(512 * 2**20 // per_row)))
    for y0 in range(0, H, rows):
        y1 = min(H, y0 + rows)
        idx = (ry[y0:y1, None, :, None] * w + rx[None, :, None, :]).reshape(-1)
        kn = kf.index_select(1, idx).view(B, y1 - y0, W, K, N, D)
        vn = vf.index_select(1, idx).view(B, y1 - y0, W, K, N, Dv)
        att = (torch.einsum("brwnd,brwknd->brwnk", query[:, y0:y1].float(), kn) * scale).softmax(-1)
        r = torch.einsum("brwnk,brwknd->brwnd", att, vn)            # [B, r, W, N, Dv]
        out[:, :, y0:y1] = r.permute(0, 3, 4, 1, 2).flatten(1, 2).to(out_dtype)
    return out
