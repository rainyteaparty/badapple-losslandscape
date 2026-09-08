"""Re-blend cached terms into final heightmaps. Cheap: no network forward pass.

w1 weights the gradient (loss-landscape) term -- ReLU-gated edges and ridges.
w2 weights the activation term -- the silhouette fill that makes it legible.
"""
import argparse, os, subprocess
import numpy as np
import frames as FR, heights as H


def build(loss, w1=1.0, w2=1.0, ema=0.6, blur=0.8, tag=None, write_mp4=True,
          write_npy=None):
    zg = np.load(FR.out(f"cache_grad_{loss}.npy")).astype(np.float32) / 255.0
    za = np.load(FR.out("cache_act.npy")).astype(np.float32) / 255.0
    z = H.norm_global(w1 * zg + w2 * za)
    z = H.temporal_ema(H.gaussian_blur(z, blur), ema)
    z = H.norm_global(z, 0.5, 99.5)
    tag = tag or f"{loss}_w2-{w2:g}"
    if write_npy is None:
        write_npy = write_mp4          # sweeps pass write_mp4=False and want neither
    if write_npy:
        np.save(FR.out(f"heights_{tag}.npy"), z.astype(np.float16))
    if write_mp4:
        mp4 = FR.out(f"heights_{tag}.mp4")
        p = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray",
             "-s", f"{z.shape[2]}x{z.shape[1]}", "-r", "30", "-i", "-",
             "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", mp4], stdin=subprocess.PIPE)
        p.stdin.write((z * 255).round().astype(np.uint8).tobytes())
        p.stdin.close(); p.wait()
        print(f"  {tag}: {os.path.getsize(mp4)/1e6:.2f} MB")
    return z


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", default="entropy")
    ap.add_argument("--w1", type=float, default=1.0)
    ap.add_argument("--w2", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.6)
    ap.add_argument("--blur", type=float, default=0.8)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    build(a.loss, a.w1, a.w2, a.ema, a.blur, a.tag)
