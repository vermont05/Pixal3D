# Pixal3D on an 8 GB AMD GPU (ROCm)

Pixal3D image to textured GLB on an RX 7600M XT (gfx1102, 8 GB) eGPU with a Ryzen 6800U host (30 GB RAM), using
TheRock ROCm 10.1. The 1024 pipeline (`1024_cascade`) takes 9.6 min, with 7.28 GB peak VRAM and 3.4 GB of RAM left.

The Python changes are committed on this branch (see `git diff f7cf384`). This folder holds everything else.

## Layout

The scripts expect this layout:

```
~/src/pixal3d/
  .venv/              Python venv (PyTorch for ROCm)
  Pixal3D/            this repo, branch rocm-8gb
  TRELLIS.2/          https://github.com/microsoft/TRELLIS.2 (for o-voxel)
  build/CuMesh/       https://github.com/visualbruno/CuMesh + patches/cumesh-hip.patch
  shims/              copy of rocm/shims
~/opt/therock-10.1.0/install   TheRock ROCm (hipcc, headers)
~/models/pixal3d               model weights
```

## Files

| Path | What it is |
|---|---|
| `scripts/run_pixal3d.sh` | Runs one image. Sets the required environment (below) |
| `scripts/guarded.sh` | `run_pixal3d.sh` with a RAM guard, logging RAM and VRAM every 5 s |
| `scripts/build_env.sh`, `scripts/build_ext.sh` | Build CuMesh, FlexGEMM and o-voxel as HIP extensions for gfx1102 |
| `scripts/check_ext.py` | Checks the built extensions against PyTorch references |
| `shims/natten` | Plain-PyTorch neighborhood attention in place of NATTEN, plus `na2d_lowres`, which skips the 4 GB upsampled key/value maps |
| `shims/nvdiffrast` | Plain-PyTorch UV rasterize and interpolate, all that the texture bake uses |
| `shims/sitecustomize.py` | Optional stall watchdog (`PIXAL_FH=<file>`) |
| `patches/cumesh-hip.patch` | CuMesh HIP port (hipified sources, rocprim tuple, TheRock includes), against visualbruno/CuMesh |
| `patches/naf-lowres.patch` | NAF calls `na2d_lowres`. Apply it to the torch hub copy, `~/.cache/torch/hub/valeoai_NAF_main` |
| `mesh-studio/` | Small FastAPI web UI. Each job runs `run_pixal3d.sh` as a subprocess, so the UI holds no VRAM. Start it with `start-mesh.sh` (port 8090) |

## Required settings (all in run_pixal3d.sh)

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`. Upstream forced it on. With it on, one run read back a singular
  matrix after a failed mapping, and another stalled single-threaded for 2 hours. Both stopped once it was off.
- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` for memory-efficient SDPA: 20k tokens need 0.25 GB instead of 17.9 GB.
- `PIXAL_ONESHOT=1` frees each model once its stage is done. Without it RAM goes over 23 GB.
- `ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa`, `ROCR_VISIBLE_DEVICES=1` (only the eGPU), and no `HSA_OVERRIDE_GFX_VERSION`.

## Known limits

- Input images need an alpha channel. Background removal is not set up, because `briaai/RMBG-2.0` is gated.
- Only `1024_cascade` and `1536_cascade` are supported in projection mode. 1536 has not been tried on 8 GB.
