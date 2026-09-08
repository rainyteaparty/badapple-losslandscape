"""Self-supervised pretext task so dL/da means something.

A randomly initialised head gives gradients dominated by the head's own random
structure rather than by image content -- the landscape comes out as noise.
Rotation prediction (Gidaris et al. 2018) forces genuine shape sensitivity,
needs no labels and no downloads, and trains in under a minute.
"""
import numpy as np, torch, torch.nn.functional as F
from net import MiniAlex


def train_rotation(frames, device, steps=400, batch=16, lr=3e-4, seed=0,
                   logit_scale=1.0, pool="avg", log_every=100):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MiniAlex(num_classes=4, logit_scale=logit_scale, pool=pool,
                     seed=seed).to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for step in range(steps):
        idx = rng.integers(0, len(frames), batch)
        x = torch.from_numpy(frames[idx]).unsqueeze(1)
        k = torch.from_numpy(rng.integers(0, 4, batch))
        x = torch.stack([torch.rot90(x[i], int(k[i]), (1, 2)) for i in range(batch)]).to(device)
        loss = F.cross_entropy(model(x), k.to(device))
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(loss.item())
        if log_every and (step + 1) % log_every == 0:
            print(f"  step {step+1:>4}  loss {np.mean(hist[-log_every:]):.4f}")
    return model.eval(), hist


def rotation_acc(model, frames, device, n=64, seed=1):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(frames), n)
    x = torch.from_numpy(frames[idx]).unsqueeze(1)
    k = torch.from_numpy(rng.integers(0, 4, n))
    x = torch.stack([torch.rot90(x[i], int(k[i]), (1, 2)) for i in range(n)]).to(device)
    with torch.no_grad():
        return (model(x).argmax(-1).cpu() == k).float().mean().item()
