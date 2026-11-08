# Source this before building HIP extensions for the Pixal3D venv (TheRock 10.1 hipcc, gfx1102 eGPU).
R=$HOME/opt/therock-10.1.0/install
export ROCM_HOME=$R ROCM_PATH=$R HIP_PATH=$R HIP_HOME=$R CUDA_HOME=$R
export PYTORCH_ROCM_ARCH=gfx1102 HCC_AMDGPU_TARGET=gfx1102 AMDGPU_TARGETS=gfx1102 FORCE_CUDA=1 MAX_JOBS=12
# hipcc otherwise searches /usr/include/hip (an older system HIP) before TheRock's headers -> __ocml_* errors
export CFLAGS="-isystem $R/include" CXXFLAGS="-isystem $R/include" HIPFLAGS="-isystem $R/include"
export PATH=$R/bin:$R/lib/llvm/bin:$HOME/src/pixal3d/.venv/bin:$PATH
export VIRTUAL_ENV=$HOME/src/pixal3d/.venv
export ROCR_VISIBLE_DEVICES=1 ROCPROFILER_REGISTER_ENABLED=0
unset HSA_OVERRIDE_GFX_VERSION
