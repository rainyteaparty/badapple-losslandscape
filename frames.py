"""Frame sources: real video via ffmpeg, or a synthetic silhouette clip for testing."""
import subprocess, math, pathlib
import numpy as np

SIZE = 448

# All generated artifacts (frames, heightmaps, previews, encoded video) live
# alongside the source clip in video/, which is gitignored wholesale.
VIDEO_DIR = pathlib.Path(__file__).resolve().parent / "video"
# Generated artifacts go in a subdirectory, NOT next to the source clip: the
# source scan globs video/*.mp4 and takes the first match, so writing our own
# mp4s alongside it makes the pipeline read its own output as input.
OUT_DIR = VIDEO_DIR / "out"


def find_video():
    for ext in ("*.mp4", "*.mkv", "*.webm", "*.mov", "*.avi"):
        hits = sorted(VIDEO_DIR.glob(ext))       # non-recursive: skips out/
        if hits:
            return str(hits[0])
    raise FileNotFoundError(f"no source video in {VIDEO_DIR}")


def out(name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return str(OUT_DIR / name)


def _letterbox(frame_hw, size=SIZE):
    """Fit a HxW float array into a square canvas without stretching. Bad Apple is 4:3."""
    h, w = frame_hw.shape
    s = min(size / h, size / w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    yi = (np.arange(nh) / s).astype(np.int64).clip(0, h - 1)
    xi = (np.arange(nw) / s).astype(np.int64).clip(0, w - 1)
    small = frame_hw[yi][:, xi]
    canvas = np.zeros((size, size), np.float32)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = small
    return canvas


def from_video(path, size=SIZE, stride=1, limit=None):
    """Decode a video to grayscale float32 [0,1] frames, letterboxed to size x size."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", path],
        capture_output=True, text=True, check=True).stdout.strip()
    w, h = (int(v) for v in probe.split("x"))
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        stdout=subprocess.PIPE)
    n, out = 0, []
    while True:
        buf = proc.stdout.read(w * h)
        if len(buf) < w * h:
            break
        if n % stride == 0:
            f = np.frombuffer(buf, np.uint8).reshape(h, w).astype(np.float32) / 255.0
            out.append(_letterbox(f, size))
            if limit and len(out) >= limit:
                break
        n += 1
    proc.stdout.close()
    if proc.poll() is None:      # early --limit exit: kill rather than SIGPIPE-spam
        proc.kill()
    proc.wait()
    return np.stack(out)