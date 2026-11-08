# Pure-PyTorch stand-in for the parts of nvdiffrast.torch that TRELLIS.2 / Pixal3D use to bake textures:
# rasterize() of 2-D triangles (UV space, w = 1) and interpolate(). nvdiffrast's CUDA rasterizer is NVIDIA-only.
# Conventions follow nvdiffrast: pixel centres at ((x + 0.5) / W * 2 - 1, (y + 0.5) / H * 2 - 1), row 0 at y = -1;
# rast[..., 0:2] = barycentrics (u, v) with p = u*v0 + v*v1 + (1-u-v)*v2, rast[..., 2] = z/w, rast[..., 3] = tri id + 1.
# No depth test between overlapping triangles (UV charts do not overlap); no gradients.
import torch


class RasterizeCudaContext:
    def __init__(self, device=None):
        self.device = device


RasterizeGLContext = RasterizeCudaContext

_MAX_CANDIDATES = 1 << 25   # candidate pixels per pass (~32M), bounds memory


@torch.no_grad()
def rasterize(glctx, pos, tri, resolution, ranges=None, grad_db=True):
    H, W = int(resolution[0]), int(resolution[1])
    if pos.dim() == 2:
        pos = pos[None]
    assert pos.shape[0] == 1 and ranges is None, "rasterize shim: single minibatch only"
    dev = pos.device
    p = pos[0].float()
    tri = tri.long()
    xy = p[:, :2] / p[:, 3:4]
    z = p[:, 2] / p[:, 3]
    px = (xy[:, 0] + 1) * 0.5 * W - 0.5
    py = (xy[:, 1] + 1) * 0.5 * H - 0.5
    rast = torch.zeros(1, H, W, 4, device=dev, dtype=torch.float32)
    tx, ty = px[tri], py[tri]                                   # [F, 3]
    area = (tx[:, 1] - tx[:, 0]) * (ty[:, 2] - ty[:, 0]) - (tx[:, 2] - tx[:, 0]) * (ty[:, 1] - ty[:, 0])
    x0 = tx.min(1).values.ceil().clamp(0, W - 1).long(); x1 = tx.max(1).values.floor().clamp(-1, W - 1).long()
    y0 = ty.min(1).values.ceil().clamp(0, H - 1).long(); y1 = ty.max(1).values.floor().clamp(-1, H - 1).long()
    bw = (x1 - x0 + 1).clamp(min=0); bh = (y1 - y0 + 1).clamp(min=0)
    keep = (area.abs() > 1e-12) & (bw > 0) & (bh > 0)
    fid = keep.nonzero().squeeze(1)
    n = (bw * bh)[fid]
    csum = torch.cumsum(n, 0)
    start = 0
    while start < fid.numel():
        base = int(csum[start - 1]) if start > 0 else 0
        end = int(torch.searchsorted(csum, base + _MAX_CANDIDATES, right=True))
        end = max(end, start + 1)
        f = fid[start:end]; cnt = n[start:end]
        owner = torch.repeat_interleave(torch.arange(f.numel(), device=dev), cnt)
        offs = torch.arange(int(cnt.sum()), device=dev) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
        fo = f[owner]
        X = x0[fo] + offs % bw[fo]
        Y = y0[fo] + offs // bw[fo]
        cx, cy = X.float(), Y.float()
        ax, ay = tx[fo], ty[fo]
        a = area[fo]
        b0 = ((ax[:, 1] - cx) * (ay[:, 2] - cy) - (ax[:, 2] - cx) * (ay[:, 1] - cy)) / a
        b1 = ((ax[:, 2] - cx) * (ay[:, 0] - cy) - (ax[:, 0] - cx) * (ay[:, 2] - cy)) / a
        b2 = 1 - b0 - b1
        eps = -1e-6
        inside = (b0 >= eps) & (b1 >= eps) & (b2 >= eps)
        X, Y, b0, b1, fo = X[inside], Y[inside], b0[inside], b1[inside], fo[inside]
        zz = b0 * z[tri[fo, 0]] + b1 * z[tri[fo, 1]] + (1 - b0 - b1) * z[tri[fo, 2]]
        rast[0, Y, X] = torch.stack([b0, b1, zz, (fo + 1).float()], dim=-1)
        start = end
    return rast, torch.zeros_like(rast) if grad_db else None


@torch.no_grad()
def interpolate(attr, rast, tri, rast_db=None, diff_attrs=None):
    a = attr[0] if attr.dim() == 3 else attr                    # [V, C]
    tid = rast[..., 3].long() - 1                               # [B, H, W]
    m = tid >= 0
    out = torch.zeros(*rast.shape[:-1], a.shape[-1], device=a.device, dtype=a.dtype)
    t = tri.long()[tid[m]]                                      # [P, 3]
    u = rast[..., 0][m][:, None].to(a.dtype); v = rast[..., 1][m][:, None].to(a.dtype)
    out[m] = u * a[t[:, 0]] + v * a[t[:, 1]] + (1 - u - v) * a[t[:, 2]]
    return out, None
