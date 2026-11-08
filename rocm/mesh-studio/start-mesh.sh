#!/usr/bin/env bash
# em680: Mesh Studio (Pixal3D image to textured GLB on the eGPU), http://0.0.0.0:8090
# The studio holds no VRAM. Each job is a subprocess, ~/src/pixal3d/run_pixal3d.sh, which sets the GPU environment.
MESHGEN_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$MESHGEN_DIR/logs"
export PYTHONUNBUFFERED=1 MESHGEN_DIR
exec "$HOME/src/pixal3d/.venv/bin/python" -m uvicorn app:app --app-dir "$MESHGEN_DIR/studio" --host 0.0.0.0 --port "${MESH_PORT:-8090}" >> "$MESHGEN_DIR/logs/mesh.log" 2>&1
