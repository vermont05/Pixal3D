import os
import re
import io
import glob
import time
import json
import signal
import threading
import subprocess
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

try:
    from PIL import Image
except Exception:  # pragma: no cover - pillow is expected to be present
    Image = None


BASE = Path(os.environ.get("MESHGEN_DIR", os.path.expanduser("~/meshgen")))
UPLOADS = BASE / "uploads"
OUTPUTS = BASE / "outputs"
STATIC = Path(__file__).resolve().parent / "static"

# Path of the run script; override with MESH_RUN_SCRIPT for testing.
RUN_SCRIPT = os.environ.get("MESH_RUN_SCRIPT", os.path.expanduser("~/src/pixal3d/run_pixal3d.sh"))

UPLOADS.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

RAM_GUARD_MB = 1500
ID_RE = re.compile(r"^[0-9A-Za-z_-]+\Z")
TQDM_RE = re.compile(r"(\d+)%\|.*?\| *(\d+/\d+)")

# Ordered stage substrings -> display labels. Order matters for matching.
STAGE_SUBS = [
    ("[Done] GLB saved", "done"),
    ("Extracting GLB", "GLB export"),
    ("Sampling sparse structure", "Sparse structure"),
    ("Sampling HR shape SLat", "Shape 1024"),
    ("Sampling shape SLat", "Shape 512"),
    ("Sampling texture SLat", "Texture"),
]
STAGE_ORDER = [
    "Loading models",
    "Sparse structure",
    "Shape 512",
    "Shape 1024",
    "Texture",
    "GLB export",
]


def _read_int(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return None


def _find_amdgpu_card():
    for card in glob.glob("/sys/class/drm/card[0-9]*/device"):
        vendor = _read_int_safe(card + "/vendor")
        device = _read_int_safe(card + "/device")
        if vendor == "0x1002" and device == "0x7480":
            return card
    return None


def _read_int_safe(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return None


def _read_vram():
    card = _VRAM_CARD
    if not card:
        return 0.0, 0.0
    used = _read_int(card + "/mem_info_vram_used")
    total = _read_int(card + "/mem_info_vram_total")
    used = (used or 0) / 1048576.0
    total = (total or 0) / 1048576.0
    return used, total


_VRAM_CARD = _find_amdgpu_card()


def _read_ram_avail_kb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return None


def _read_ram_total_mb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def parse_log(text, stage_times):
    """Parse tqdm/progress text. Mutates stage_times. Returns (stage, percent, step, detail)."""
    lines = [p for p in re.split(r"[\r\n]", text) if p.strip() != ""]
    stage = "Loading models"
    percent = 0
    step = ""
    detail = ""
    for line in lines:
        matched = None
        for sub, label in STAGE_SUBS:
            if sub in line:
                matched = label
                break
        if matched and matched != "done":
            if stage != matched:
                stage = matched
        elif matched == "done":
            stage = "GLB export"
        m = TQDM_RE.search(line)
        if m:
            percent = int(m.group(1))
            step = m.group(2)
            label = line[:m.start()].strip(" :")
            if stage == "GLB export":
                detail = label
    return stage, percent, step, detail


def clean_log_lines(text):
    """Keep only the final state of each \\r-rewritten line; drop lines containing 'Warning'."""
    out = []
    for raw in text.split("\n"):
        # tqdm rewrites a line with \r; the final state is the segment after the last \r.
        line = raw.split("\r")[-1].strip()
        if not line:
            continue
        if "Warning" in line:
            continue
        out.append(line)
    return out[-40:]


class Job:
    def __init__(self, id, seed, source):
        self.id = id
        self.seed = seed
        self.source = source
        self.status = "queued"
        self.error = None
        self.started = None
        self.finished = None
        self.stage = "Loading models"
        self.percent = 0
        self.step = ""
        self.detail = ""
        self.stage_times = {}
        self.peak_vram_mb = 0.0
        self.min_ram_mb = None
        self.glb_size = 0
        self._proc = None
        self._stage_started = None

    @property
    def elapsed(self):
        if self.started is None:
            return 0.0
        end = self.finished if self.finished is not None else time.time()
        return max(0.0, end - self.started)

    def paths(self):
        return (
            UPLOADS / f"{self.id}.png",
            OUTPUTS / f"{self.id}.glb",
            OUTPUTS / f"{self.id}.log",
        )

    def _track_stage(self, new_stage):
        now = time.time()
        if self._stage_started is not None:
            key = self.stage if self.stage in STAGE_ORDER else "Loading models"
            self.stage_times[key] = self.stage_times.get(key, 0.0) + (now - self._stage_started)
        self.stage = new_stage if new_stage in STAGE_ORDER else "Loading models"
        self._stage_started = now

    def _finish(self, status, error=None):
        now = time.time()
        if self._stage_started is not None:
            key = self.stage if self.stage in STAGE_ORDER else "Loading models"
            self.stage_times[key] = self.stage_times.get(key, 0.0) + (now - self._stage_started)
            self._stage_started = None
        self.finished = now
        self.status = status
        self.error = error
        glb = OUTPUTS / f"{self.id}.glb"
        if glb.exists():
            self.glb_size = glb.stat().st_size
        self._write_json()

    def _write_json(self):
        data = {
            "id": self.id,
            "seed": self.seed,
            "status": self.status,
            "error": self.error,
            "started": self.started,
            "finished": self.finished,
            "elapsed": self.elapsed,
            "stage_times": self.stage_times,
            "peak_vram_mb": self.peak_vram_mb,
            "min_ram_mb": self.min_ram_mb,
            "source": self.source,
            "glb_size": self.glb_size,
        }
        try:
            with open(OUTPUTS / f"{self.id}.json", "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def to_dict(self):
        return {
            "id": self.id,
            "seed": self.seed,
            "status": self.status,
            "error": self.error,
            "started": self.started,
            "finished": self.finished,
            "elapsed": self.elapsed,
            "stage": self.stage,
            "percent": self.percent,
            "step": self.step,
            "detail": self.detail,
            "stage_times": self.stage_times,
            "peak_vram_mb": self.peak_vram_mb,
            "min_ram_mb": self.min_ram_mb,
            "source": self.source,
            "glb_size": self.glb_size,
        }


class Studio:
    def __init__(self):
        self.jobs = {}  # id -> Job
        self.queue = []  # list of ids, FIFO
        self.running_id = None
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.shutdown = False
        self._worker = threading.Thread(target=self._work, daemon=True)
        self._worker.start()

    # -- lifecycle -------------------------------------------------------
    def reload(self):
        # Finished jobs: reload from outputs/*.json, newest first.
        loaded = set()
        for p in sorted(OUTPUTS.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                data = json.loads(p.read_text())
            except Exception:
                continue
            jid = data.get("id")
            if not jid or not ID_RE.match(jid or ""):
                continue
            job = Job(jid, data.get("seed", 42), data.get("source", ""))
            job.status = data.get("status", "done")
            job.error = data.get("error")
            job.started = data.get("started")
            job.finished = data.get("finished")
            job.stage_times = data.get("stage_times", {})
            job.peak_vram_mb = data.get("peak_vram_mb", 0.0)
            job.min_ram_mb = data.get("min_ram_mb")
            job.glb_size = data.get("glb_size", 0)
            # A terminal stage
            if job.status == "done":
                job.stage = "GLB export"
            self.jobs[jid] = job
            loaded.add(jid)
        # Jobs that were running when the studio died: upload exists, no json.
        for p in UPLOADS.glob("*.png"):
            jid = p.stem
            if not ID_RE.match(jid):
                continue
            if jid in loaded:
                continue
            if (OUTPUTS / f"{jid}.json").exists():
                continue
            job = Job(jid, 42, p.name)
            job.status = "failed"
            job.error = "studio restarted"
            job.started = p.stat().st_mtime
            job.finished = time.time()
            self.jobs[jid] = job

    # -- queue -----------------------------------------------------------
    def enqueue(self, job):
        with self.lock:
            self.queue.append(job.id)
        self.wake.set()

    def _work(self):
        while not self.shutdown:
            job = None
            with self.lock:
                if self.running_id is None and self.queue:
                    jid = self.queue.pop(0)
                    self.running_id = jid
                    job = self.jobs.get(jid)
            if job is None:
                self.wake.wait(timeout=1.0)
                self.wake.clear()
                continue
            try:
                self._run(job)
            finally:
                with self.lock:
                    self.running_id = None
                self.wake.set()

    def _run(self, job):
        upload_path, glb_path, log_path = job.paths()
        job.status = "running"
        job.started = time.time()
        job.stage = "Loading models"
        job._stage_started = time.time()
        job.stage_times = {}
        job.peak_vram_mb = 0.0
        job.min_ram_mb = None

        if not upload_path.exists():
            job._finish("failed", "upload file missing")
            return

        try:
            with open(log_path, "w") as logf:
                cmd = [RUN_SCRIPT, str(upload_path), str(glb_path), "1024", str(job.seed)]
                proc = subprocess.Popen(
                    cmd,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                job._proc = proc
                pgid = proc.pid
                peak_vram = 0.0
                min_ram = None
                last_check = 0.0
                while True:
                    # Honor an external terminal state (e.g. a cancel request).
                    if job.status in ("cancelled", "failed"):
                        break
                    time.sleep(2)
                    if proc.poll() is not None:
                        break
                    now = time.time()
                    if now - last_check >= 2:
                        last_check = now
                        ram_kb = _read_ram_avail_kb()
                        if ram_kb is not None:
                            ram_mb = ram_kb / 1024.0
                            min_ram = ram_mb if min_ram is None else min(min_ram, ram_mb)
                            if ram_mb < RAM_GUARD_MB:
                                self._kill_group(pgid, proc)
                                job._finish("failed", "stopped: system RAM ran low")
                                return
                        vu, vt = _read_vram()
                        peak_vram = max(peak_vram, vu)
                    # parse progress
                    text = self._tail(log_path)
                    stage, percent, step, detail = parse_log(text, job.stage_times)
                    job._track_stage(stage)
                    job.percent = percent
                    job.step = step
                    job.detail = detail
                    job.peak_vram_mb = peak_vram
                    job.min_ram_mb = min_ram
                # process exited (or was cancelled / stopped externally)
                job.peak_vram_mb = peak_vram
                job.min_ram_mb = min_ram
                if job.status in ("cancelled", "failed", "done"):
                    if proc.poll() is None:
                        self._kill_group(proc.pid, proc)
                    if job.status == "cancelled" and job.finished is not None:
                        job._write_json()
                    return
                rc = proc.returncode
                text = self._tail(log_path)
                stage, percent, step, detail = parse_log(text, job.stage_times)
                job._track_stage(stage)
                job.percent = percent
                job.step = step
                job.detail = detail
                if rc != 0 or not glb_path.exists():
                    job._finish("failed", self._fail_message(text))
                else:
                    job._finish("done")
        except Exception as e:  # pragma: no cover
            job._finish("failed", f"run error: {e}")

    @staticmethod
    def _tail(path, n=200000):
        try:
            size = path.stat().st_size
            with open(path, "rb") as f:
                if size > n:
                    f.seek(size - n)
                data = f.read()
            return data.decode("utf-8", "replace")
        except Exception:
            return ""

    def _fail_message(self, text):
        lines = [l for l in re.split(r"[\r\n]", text) if l.strip() != ""]
        if any("out of memory" in l.lower() for l in lines):
            return "out of memory"
        # last non-progress line containing "Error"
        err = None
        for l in lines:
            if "Error" in l and not TQDM_RE.search(l):
                err = l.strip()
        if err:
            return err
        if lines:
            return lines[-1].strip()
        return "process failed"

    @staticmethod
    def _kill_group(pgid, proc):
        try:
            os.killpg(pgid, signal.SIGTERM)
        except Exception:
            pass
        deadline = time.time() + 5
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.2)
        if proc.poll() is None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except Exception:
                pass

    # -- operations ------------------------------------------------------
    def create(self, seed, source_name, image_bytes):
        jid = time.strftime("%Y%m%d-%H%M%S") + "-" + _rand4()
        while jid in self.jobs:
            jid = time.strftime("%Y%m%d-%H%M%S") + "-" + _rand4()
        job = Job(jid, seed, source_name)
        # Write the upload before the job becomes visible so the worker never races it.
        (UPLOADS / f"{jid}.png").write_bytes(image_bytes)
        with self.lock:
            self.jobs[jid] = job
        self.enqueue(job)
        return job

    def cancel(self, jid):
        with self.lock:
            if jid in self.queue:
                self.queue.remove(jid)
                job = self.jobs.get(jid)
                if job:
                    job.status = "cancelled"
                    job.finished = time.time()
                    job.error = "cancelled"
                    job._write_json()
                return True
            if jid == self.running_id:
                job = self.jobs.get(jid)
                if job:
                    job.status = "cancelled"
                    job.finished = time.time()
                    job.error = "cancelled"
                    # The worker thread finalizes (writes json) once it observes the
                    # process is gone, so we only signal + kill here to avoid a race.
                    if job._proc is not None and job._proc.poll() is None:
                        self._kill_group(job._proc.pid, job._proc)
                return True
        return False

    def delete(self, jid):
        with self.lock:
            if jid == self.running_id or jid in self.queue:
                return False
            job = self.jobs.pop(jid, None)
        for suffix in (".png",):
            p = UPLOADS / f"{jid}{suffix}"
            if p.exists():
                p.unlink()
        for suffix in (".glb", ".log", ".json"):
            p = OUTPUTS / f"{jid}{suffix}"
            if p.exists():
                p.unlink()
        return True

    def status(self):
        with self.lock:
            running = self.jobs[self.running_id].to_dict() if self.running_id else None
            queued = [self.jobs[i].to_dict() for i in self.queue if i in self.jobs]
        vu, vt = _read_vram()
        rk = _read_ram_avail_kb()
        return {
            "ok": True,
            "running": running,
            "queued": queued,
            "vram_used_mb": round(vu, 1),
            "vram_total_mb": round(vt, 1),
            "ram_available_mb": round((rk or 0) / 1024.0, 1),
            "ram_total_mb": round(_read_ram_total_mb(), 1),
        }

    def jobs_list(self):
        with self.lock:
            items = list(self.jobs.values())
        items.sort(key=lambda j: (j.started or 0), reverse=True)
        return [j.to_dict() for j in items]

    def log(self, jid):
        p = OUTPUTS / f"{jid}.log"
        if not p.exists():
            return ""
        return "\n".join(clean_log_lines(p.read_text()))

    def shutdown_jobs(self):
        self.shutdown = True
        with self.lock:
            jid = self.running_id
            proc = self.jobs[jid]._proc if jid and jid in self.jobs else None
        if proc is not None and proc.poll() is None:
            self._kill_group(proc.pid, proc)
        self.wake.set()


def _rand4():
    import secrets
    return secrets.token_hex(2)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
from contextlib import asynccontextmanager  # noqa: E402

studio = Studio()


@asynccontextmanager
async def lifespan(_app):
    studio.reload()
    yield
    studio.shutdown_jobs()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def index():
    p = STATIC / "index.html"
    return HTMLResponse(p.read_text())


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/health")
def health():
    return {"ok": True}


def _alpha_check(img):
    """Return (has_alpha_and_transparency, message). message is set when rejected."""
    if Image is None:
        return True, None
    if img.mode not in ("RGBA", "LA", "PA"):
        # Also handle "P" with transparency
        if img.mode == "P" and "transparency" in img.info:
            img = img.convert("RGBA")
        else:
            return False, "needs a transparent background; automatic background removal is not set up"
    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    # fully opaque?
    extreme = alpha.getextrema()
    if extreme and extreme[0] == 255:
        return False, "needs a transparent background; automatic background removal is not set up"
    return True, None


@app.post("/api/jobs")
async def create_job(
    image: UploadFile = File(...),
    seed: int = Form(42),
):
    data = await image.read()
    if not data:
        return JSONResponse({"detail": "empty image"}, status_code=400)
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:
        return JSONResponse({"detail": f"could not read image: {e}"}, status_code=400)
    ok, msg = _alpha_check(img)
    if not ok:
        return JSONResponse({"detail": msg}, status_code=400)
    # Save as PNG (preserves alpha)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png = buf.getvalue()
    name = image.filename or "upload.png"
    job = studio.create(seed, name, png)
    return {"ok": True, "id": job.id}


@app.get("/api/status")
def status():
    return studio.status()


@app.get("/api/jobs")
def list_jobs():
    return studio.jobs_list()


@app.get("/api/jobs/{id}/log")
def job_log(id: str):
    if not ID_RE.match(id):
        return JSONResponse({"detail": "invalid id"}, status_code=400)
    return PlainTextResponse(studio.log(id))


@app.post("/api/jobs/{id}/cancel")
def job_cancel(id: str):
    if not ID_RE.match(id):
        return JSONResponse({"detail": "invalid id"}, status_code=400)
    if not studio.cancel(id):
        return JSONResponse({"detail": "not found"}, status_code=404)
    return {"ok": True}


@app.delete("/api/jobs/{id}")
def job_delete(id: str):
    if not ID_RE.match(id):
        return JSONResponse({"detail": "invalid id"}, status_code=400)
    if not studio.delete(id):
        return JSONResponse({"detail": "job is running or not found"}, status_code=409)
    return {"ok": True}


@app.get("/files/{id}.glb")
def file_glb(id: str):
    if not ID_RE.match(id):
        return JSONResponse({"detail": "invalid id"}, status_code=400)
    p = OUTPUTS / f"{id}.glb"
    if not p.exists():
        return JSONResponse({"detail": "not found"}, status_code=404)
    return FileResponse(str(p), media_type="model/gltf-binary")


@app.get("/files/{id}.png")
def file_png(id: str):
    if not ID_RE.match(id):
        return JSONResponse({"detail": "invalid id"}, status_code=400)
    p = UPLOADS / f"{id}.png"
    if not p.exists():
        return JSONResponse({"detail": "not found"}, status_code=404)
    return FileResponse(str(p), media_type="image/png")
