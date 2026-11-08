#!/bin/bash
# Builds CuMesh, FlexGEMM and o-voxel as HIP extensions (gfx1102, TheRock 10.1 hipcc) into ~/src/pixal3d/.venv.
# Patches follow egore/comfyui-trellis2-gguf-rocm (written on a 7600 XT). CuMesh is already cloned and patched.
set -e
source ~/src/pixal3d/build_env.sh
B=~/src/pixal3d/build; R=$ROCM_HOME; LOG=$B/build_ext.log; : > $LOG
inc="\"-isystem\", \"$R/include\""

echo "== CuMesh"; cd $B/CuMesh
GPU_ARCHS=gfx1102 uv pip install --no-build-isolation --no-deps . >> $LOG 2>&1 || { echo "CuMesh FAILED, see $LOG"; exit 1; }
SP=$(python -c "import site;print(site.getsitepackages()[0])")
curl -sfL -o $SP/cumesh/remeshing.py https://raw.githubusercontent.com/visualbruno/CuMesh/main/cumesh/remeshing.py

echo "== FlexGEMM"; cd $B; [ -d FlexGEMM ] || git clone -q --recursive https://github.com/JeffreyXiang/FlexGEMM.git; cd FlexGEMM
grep -q isystem setup.py || sed -i "s/-std=c++17/-std=c++20/g; s|\"nvcc\": \[|\"nvcc\": [$inc, \"-std=c++20\", |; s|\"cxx\": \[|\"cxx\": [$inc, \"-std=c++20\", |" setup.py
uv pip install --no-build-isolation --no-deps . >> $LOG 2>&1 || { echo "FlexGEMM FAILED, see $LOG"; exit 1; }
CFG=$SP/flex_gemm/kernels/triton/spconv/config.py
grep -q 'torch.version, "hip"' $CFG || { sed -i '1s/^/import torch\n/' $CFG; sed -i 's/^allow_tf32 = True$/allow_tf32 = not getattr(torch.version, "hip", None)  # TF32 is NVIDIA-only/' $CFG; }

echo "== o-voxel"; git -C ~/src/pixal3d/TRELLIS.2 submodule update --init --recursive >> $LOG 2>&1
rm -rf $B/o-voxel; cp -r ~/src/pixal3d/TRELLIS.2/o-voxel $B/o-voxel; cd $B/o-voxel
sed -i '/cumesh.*git+/d' pyproject.toml
grep -q isystem setup.py || sed -i "s/-std=c++17/-std=c++20/g; s|\"nvcc\": \[|\"nvcc\": [$inc, \"-std=c++20\", |; s|\"cxx\": \[|\"cxx\": [$inc, \"-std=c++20\", |" setup.py
uv pip install --no-build-isolation --no-deps . >> $LOG 2>&1 || { echo "o-voxel FAILED, see $LOG"; exit 1; }

cd ~; python -c "import torch, cumesh, flex_gemm, o_voxel; print('ALL IMPORT OK')"
