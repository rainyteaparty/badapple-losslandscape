"""Full pipeline: video -> per-frame loss landscape heightmaps -> .npy + grayscale mp4.

Streams the clip in chunks; holding all 6572 frames at 448x448 float32 would be
~5.3 GB. Heightmaps themselves are small enough to keep in RAM as float16.
"""
import argparse, subprocess, time
import numpy as np, torch

import frames as FR, heights as H
from train import train_rotation, rotation_acc
from net import pick_device

# conv1 is Conv2d(k=11, s=4, p=2), so output row o covers input rows [4o-2, 4o+8].
# Letterboxing 480x360 into 448 leaves content in rows 56..391 -> output rows 14..97.
# Cropping to that removes the artificial edge the black padding would otherwise
# create, and yields a 4:3 surface. Width padded 111 -> 112 for yuv420p.
CROP_R0, CROP_R1, GRID_W = 14, 98, 112


def stream(path, size=FR.SIZE, chunk=64, stride=1, limit=None):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", path],
        capture_output=True, text=True, check=True).stdout.strip()
    w, h = (int(v) for v in probe.split("x"))
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        stdout=subprocess.PIPE)
    buf, n, total = [], 0, 0
    while True:
        raw = proc.stdout.read(w * h)
        if len(raw) < w * h:
            break
        if n % stride == 0:
            f = np.frombuffer(raw, np.uint8).reshape(h, w).astype(np.float32) / 255.0
            buf.append(FR._letterbox(f, size)); total += 1
            if len(buf) == chunk:
                yield np.stack(buf); buf = []
            if limit and total >= limit:
                break
        n += 1
    if buf:
        yield np.stack(buf)
    proc.stdout.close()
    if proc.poll() is None:      # early --limit exit: kill rather than SIGPIPE-spam
        proc.kill()
    proc.wait()


def crop(z):
    """(N,111,111) -> (N,84,112): drop letterbox rows, pad width to even."""
    z = z[:, CROP_R0:CROP_R1, :]
    return np.pad(z, ((0, 0), (0, 0), (0, GRID_W - z.shape[2])), mode="edge")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", default="entropy",
                    help="comma-separated, or 'all' for every definition in the registry")
    ap.add_argument("--w1", type=float, default=1.0, help="weight on gradient (loss) term")
    ap.add_argument("--w2", type=float, default=1.0, help="weight on activation (fill) term")
    ap.add_argument("--ema", type=float, default=0.6)
    ap.add_argument("--blur", type=float, default=0.8)
    ap.add_argument("--steps", type=int, default=3200)
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    from losses import REGISTRY
    losses = list(REGISTRY) if a.loss == "all" else a.loss.split(",")
    dev = pick_device()
    src = FR.find_video()

    print(f"[1/4] training rotation net on strided frames ({dev})")
    train_f = FR.from_video(src, stride=12)
    model, _ = train_rotation(train_f, dev, steps=a.steps, pool="avg", log_every=200)
    print(f"      rot-acc {rotation_acc(model, train_f, dev):.3f}  (chance 0.25, TRAIN split)")
    torch.save(model.state_dict(), FR.out("model.pt"))   # reused by generate_layers.py
    del train_f

    print(f"[2/4] streaming heightmaps for {len(losses)} loss definition(s)")
    G = {k: [] for k in losses}
    A, t0, n = [], time.time(), 0
    for ch in stream(src, chunk=a.chunk, limit=a.limit):
        za = None
        for k in losses:
            zg, za = H.run_all(model, ch, dev, loss=k, batch=a.chunk)
            G[k].append(crop(zg).astype(np.float16))
        A.append(crop(za).astype(np.float16))     # z_act is loss-independent
        n += len(ch)
        if n % (a.chunk * 25) == 0:
            print(f"      {n} frames  {n/(time.time()-t0):.1f} fps")
    za = np.concatenate(A).astype(np.float32)
    print(f"      {n} frames in {time.time()-t0:.1f}s -> {za.shape}")

    import os
    # Cache the per-term normalised maps as uint8. H.blend normalises each term
    # before weighting, so this is lossless w.r.t. the blend maths and lets w1/w2
    # be retuned by blend.py without re-running the 77s forward/backward sweep.
    np.save(FR.out("cache_act.npy"), (H.norm_global(za) * 255).round().astype(np.uint8))
    for k in losses:
        np.save(FR.out(f"cache_grad_{k}.npy"),
                (H.norm_global(np.concatenate(G[k]).astype(np.float32)) * 255)
                .round().astype(np.uint8))

    for k in losses:
        tag = a.tag or k
        zg = np.concatenate(G[k]).astype(np.float32)
        z = H.blend(zg, za, a.w1, a.w2)
        z = H.temporal_ema(H.gaussian_blur(z, a.blur), a.ema)
        z = H.norm_global(z, 0.5, 99.5)
        npy = FR.out(f"heights_{tag}.npy"); np.save(npy, z.astype(np.float16))
        mp4 = FR.out(f"heights_{tag}.mp4")
        u8 = (z * 255).round().astype(np.uint8)
        p = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray",
             "-s", f"{z.shape[2]}x{z.shape[1]}", "-r", "30", "-i", "-",
             "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", mp4], stdin=subprocess.PIPE)
        p.stdin.write(u8.tobytes()); p.stdin.close(); p.wait()
        print(f"      {k:>14}: {os.path.basename(mp4)} "
              f"({os.path.getsize(mp4)/1e6:.2f} MB)  npy {os.path.getsize(npy)/1e6:.1f} MB")
    print("[4/4] extracting audio for the viewer")
    au = FR.out("audio.m4a")
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-vn",
                        "-c:a", "aac", "-b:a", "128k", au], capture_output=True)
    if r.returncode == 0 and os.path.exists(au):
        print(f"      audio.m4a ({os.path.getsize(au)/1e6:.2f} MB)")
    else:
        print("      no audio track in the source clip -- viewer will be silent")
    print("      done")


if __name__ == "__main__":
    main()
