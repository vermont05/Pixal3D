#!/bin/bash
# Pixal3D image -> textured GLB on the RX 7600M XT eGPU (8 GB): low-VRAM mode, PyTorch SDPA attention,
# pure-torch NATTEN/nvdiffrast stand-ins from ~/src/pixal3d/shims. usage: run_pixal3d.sh <image> [out.glb] [resolution]
cd ~/src/pixal3d/Pixal3D
export ROCR_VISIBLE_DEVICES=1 ROCPROFILER_REGISTER_ENABLED=0 MIOPEN_FIND_MODE=FAST
unset HSA_OVERRIDE_GFX_VERSION
export ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa
export PYTHONPATH=$HOME/src/pixal3d/shims${PYTHONPATH:+:$PYTHONPATH}
export HF_HUB_OFFLINE=1
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1  # memory-efficient SDPA: 20k tokens 0.25 GB vs 17.9 GB on the math path
export PYTORCH_CUDA_ALLOC_CONF=${PIXAL_ALLOC:-expandable_segments:False}
export PIXAL_ONESHOT=1
exec ~/src/pixal3d/.venv/bin/python inference.py --image "$1" --output "${2:-$HOME/src/pixal3d/out/$(basename "${1%.*}").glb}" \
    --model_path $HOME/models/pixal3d --low_vram --resolution "${3:-1024}" --seed "${4:-42}"
