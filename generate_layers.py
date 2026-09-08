"""Per-layer loss landscapes for the whole clip, packed into one atlas mp4.

The viewer needs conv2/conv3/global-pool/fc1/prediction alongside the conv1
surface it already renders. Five separate videos would drift against each other,
so every layer is nearest-upscaled to a 128x128 tile and stacked vertically into
a single 128x640 frame -- one decoder, one clock, no sync to maintain.
"""
import os, subprocess, time
import numpy as np, torch
import frames as FR, heights as H
from net import MiniAlex, pick_device
from losses import REGISTRY
from generate import stream

W1, W2, EMA = 1.0, 1.0, 0.6      # match the main stack's shipped blend
TAPS = {2: "conv2", 5: "conv3", 7: "global pool", 9: "fc1"}


def layer_terms(model, x, loss="entropy"):
    """Both terms per layer, the way heights.frame_heights does for conv1.

    The gradient term alone tracks the model's uncertainty, and the model is
    near-certain on ~47% of frames -- so a gradient-only plate is dark for most
    of the clip. The activation term is what keeps the main surface lit, and the
    other layers need it for the same reason.
    """
    acts = {}
    a = model.pre_relu(x); h = a
    for j, L in enumerate(model.head):
        h = L(h)
        if j in TAPS:
            h.retain_grad(); acts[TAPS[j]] = h
    logits = h * model.logit_scale
    model.zero_grad(set_to_none=True)
    REGISTRY[loss](logits).sum().backward()
    G, A = {}, {}
    for k, t in acts.items():
        g, d = t.grad, t.detach()
        if g.ndim == 4:
            sc = d.flatten(2).norm(dim=2); sc = sc / (sc.mean(1, keepdim=True) + 1e-8)
            G[k] = (g * sc[:, :, None, None]).norm(dim=1).detach().cpu().numpy()
            A[k] = torch.relu(d).mean(1).cpu().numpy()
        else:
            G[k] = g.abs().detach().cpu().numpy()
            A[k] = torch.relu(d).cpu().numpy()
    P = torch.softmax(logits.detach(), -1).cpu().numpy()
    return G, A, P

TILE = 128
LAYERS = [("conv2", 55), ("conv3", 27), ("global pool", 3), ("fc1", 128), ("prediction", 4)]


def upscale(a, n=TILE):
    """Nearest-neighbour to TILExTILE. 1-D layers become vertical bars."""
    if a.ndim == 1:
        a = a[None, :]
    h, w = a.shape
    return a[(np.arange(n) * h // n)][:, (np.arange(n) * w // n)]


def main():
    dev = pick_device()
    m = MiniAlex(num_classes=4, logit_scale=1.0, pool="avg").to(dev)
    pt = FR.out("model.pt")
    if not os.path.exists(pt):
        raise SystemExit(f"{pt} missing -- run generate.py first")
    m.load_state_dict(torch.load(pt, map_location=dev)); m.eval()
    print(f"[1/3] streaming layer landscapes ({dev})")
    accG = {k: [] for k, _ in LAYERS[:-1]}
    accA = {k: [] for k, _ in LAYERS[:-1]}
    acc = {"prediction": []}
    t0, n = time.time(), 0
    for ch in stream(FR.find_video(), chunk=32):
        x = torch.from_numpy(ch).unsqueeze(1).to(dev)
        G, A, P = layer_terms(m, x)
        for k, _ in LAYERS[:-1]:
            accG[k].append(G[k].astype(np.float32)); accA[k].append(A[k].astype(np.float32))
        acc["prediction"].append(P.astype(np.float32))
        n += len(ch)
        if n % 1600 == 0:
            print(f"      {n} frames  {n/(time.time()-t0):.1f} fps")
    acc["prediction"] = np.concatenate(acc["prediction"])
    print(f"      {n} frames in {time.time()-t0:.1f}s")

    print(f"[2/3] blending w1={W1} w2={W2}, EMA {EMA}")
    for k, _ in LAYERS[:-1]:
        g = H.norm_global(np.concatenate(accG[k])); a = H.norm_global(np.concatenate(accA[k]))
        z = H.norm_global(W1*g + W2*a)
        acc[k] = H.norm_global(H.temporal_ema(z, EMA), 0.5, 99.5)
        dark = 100*(np.percentile(acc[k].reshape(n, -1), 95, axis=1) < 0.15).mean()
        print(f"      {k:<12} {str(acc[k].shape[1:]):<12} dark frames {dark:5.1f}%")
    print(f"      {'prediction':<12} {str(acc['prediction'].shape[1:]):<12} (softmax, left as-is)")

    print("[3/3] packing atlas + encoding")
    out = np.empty((n, TILE * len(LAYERS), TILE), np.uint8)
    for t, (k, _) in enumerate(LAYERS):
        for i in range(n):
            out[i, t*TILE:(t+1)*TILE] = (upscale(acc[k][i]) * 255).round().clip(0, 255)
    mp4 = FR.out("layers_atlas.mp4")
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray",
         "-s", f"{TILE}x{TILE*len(LAYERS)}", "-r", "30", "-i", "-",
         "-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv420p", mp4], stdin=subprocess.PIPE)
    p.stdin.write(out.tobytes()); p.stdin.close(); p.wait()
    print(f"      layers_atlas.mp4  {os.path.getsize(mp4)/1e6:.2f} MB  "
          f"({TILE}x{TILE*len(LAYERS)}, tiles top-to-bottom: "
          f"{', '.join(k for k,_ in LAYERS)})")


if __name__ == "__main__":
    main()
