"""FedOPAW real pilot: federated continual + open-set deepfake detection on DeepFakeFace.

Everything below is trained and evaluated for real (no hard-coded results). Scope, stated plainly:
  * data: OpenRL/DeepFakeFace, image only. Real = IMDB-WIKI faces; fakes from 3 generators
    (inpainting = SD-inpaint, insight = InsightFace swap, text2img = SD 1.5). Paired by identity.
  * leave-one-generator-out: one generator is never trained on and is the open-set family;
    the other two arrive as two sequential tasks (continual learning).
  * model "CKM-FAF-lite": ResNet-18 (ImageNet-pretrained) + radial log-spectrum branch + fusion head.
    Client and server share the architecture (no MobileViT/Swin split), so Eq.1-Eq.4 are well defined.
  * clients are simulated on one machine; no secure aggregation or DP is used.
Usage: python fedopaw_pilot.py --data data/dff_n3000_s128.npz --folds 1 2 3 --rounds 12
"""
import argparse, copy, hashlib, json, math, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import timm
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
SRC = ["wiki", "inpainting", "insight", "text2img"]  # index 0 = real
NB = 32  # radial spectrum bins

METHODS = {
    "central":        dict(),
    "fedavg":         dict(),
    "fedprox":        dict(mu=0.01),
    "fedavg_ewc":     dict(ewc="local"),
    "kd":             dict(kd=True),
    "kd_ewc":         dict(kd=True, ewc="fed"),
    "kd_ewc_gem":     dict(kd=True, ewc="fed", gem=True),
    "full":           dict(kd=True, ewc="fed", gem=True, prov=True, quant="int8"),
    "full_sparse":    dict(kd=True, ewc="fed", gem=True, prov=True, quant="sparse"),
}

# ---------------------------------------------------------------- utilities
def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    return float(roc_auc_score(y, s)) if len(set(y.tolist())) == 2 else float("nan")

def make_radial(S):
    yy, xx = torch.meshgrid(torch.arange(S), torch.arange(S), indexing="ij")
    r = torch.sqrt((yy - S // 2) ** 2 + (xx - S // 2) ** 2) / (S / 2)
    b = (r * NB).long().clamp(max=NB - 1).flatten()
    M = torch.zeros(NB, S * S); M[b, torch.arange(S * S)] = 1
    return M / M.sum(1, keepdim=True).clamp(min=1), r

class Radial:
    def __init__(self, S):
        self.M, self.r = make_radial(S)
        self.mu = 0.0; self.sd = 1.0
    def raw(self, x):  # x: B,3,S,S on CPU
        g = x.mean(1)
        P = torch.fft.fftshift(torch.fft.fft2(g), dim=(-2, -1)).abs() ** 2
        return torch.log1p(P).flatten(1) @ self.M.T
    def __call__(self, x):
        return (self.raw(x.cpu().float()) - self.mu) / self.sd

def tau_for(stem, src_idx, seed=0):
    """Synthetic C2PA trust level, deterministic per (stem, source). Correlation with label is by design."""
    h = int(hashlib.md5(f"{seed}{stem}{src_idx}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if src_idx == 0:   # authentic
        return 1.0 if h < 0.14 else 0.3 if h < 0.20 else 0.0
    return 0.7 if h < 0.20 else 1.0 if h < 0.23 else 0.3 if h < 0.28 else 0.0  # fakes: disclosed AI / forged / stripped

# ---------------------------------------------------------------- model
class Net(nn.Module):
    def __init__(self, radial):
        super().__init__()
        self.bb = timm.create_model("resnet18", pretrained=True, num_classes=0)
        self.fr = nn.Sequential(nn.Linear(NB, 64), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(512 + 64, 128), nn.ReLU())
        self.tau = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 128))
        nn.init.zeros_(self.tau[2].weight); nn.init.zeros_(self.tau[2].bias)
        self.head = nn.Linear(128, 1)
        self.radial = radial
        self.register_buffer("m", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("s", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    def feat(self, x, tau=None):
        fr = self.radial(x).to(x.device)
        h = self.fuse(torch.cat([self.bb((x - self.m) / self.s), self.fr(fr)], 1))
        if tau is not None:
            h = h + self.tau(tau.view(-1, 1))
        return h
    def forward(self, x, tau=None):
        return self.head(self.feat(x, tau)).squeeze(1)

def params(m):
    return [p for p in m.parameters() if p.requires_grad]

def flat(ts):
    return torch.cat([t.reshape(-1) for t in ts])

# ---------------------------------------------------------------- data
class Data:
    def __init__(self, path, seed):
        z = np.load(path)
        self.x = torch.from_numpy(z["x"]).permute(0, 1, 4, 2, 3).contiguous()  # N,4,3,S,S uint8
        self.stems = [str(s) for s in z["stems"]]
        self.N, _, _, self.S, _ = self.x.shape
        rng = np.random.RandomState(seed)
        perm = rng.permutation(self.N)
        a, b, c = int(.6 * self.N), int(.7 * self.N), int(.8 * self.N)
        self.train, self.val, self.proxy, self.test = perm[:a], perm[a:b], perm[b:c], perm[c:]
        self.radial = Radial(self.S)
        # standardise the spectrum branch on train reals only
        f = self.radial.raw(self.x[self.train[:400], 0].float() / 255)
        self.radial.mu = f.mean(0, keepdim=True); self.radial.sd = f.std(0, keepdim=True) + 1e-3
        self.tau = {(i, s): tau_for(self.stems[i], s, seed) for i in range(self.N) for s in range(4)}
    def batch(self, items):  # items: list of (stem_idx, src_idx)
        x = torch.stack([self.x[i, s] for i, s in items]).float() / 255
        y = torch.tensor([0. if s == 0 else 1. for _, s in items])
        t = torch.tensor([self.tau[(i, s)] for i, s in items])
        return x.to(DEV), y.to(DEV), t.to(DEV)

def partition(train, n_clients, seed, alpha=0.5):
    rng = np.random.RandomState(seed + 7)
    p = rng.dirichlet([alpha] * n_clients)
    cuts = (np.cumsum(p) * len(train)).astype(int)[:-1]
    cuts = np.clip(cuts, 1, len(train) - 1)
    parts = np.split(rng.permutation(train), np.sort(cuts))
    return [q for q in parts if len(q) >= 8][:n_clients]

def task_items(idx, fam):
    return [(int(i), 0) for i in idx] + [(int(i), fam) for i in idx]

# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def predict(model, D, items, use_tau=False, bs=128):
    model.eval(); out = []
    for k in range(0, len(items), bs):
        x, _, t = D.batch(items[k:k + bs])
        out.append(torch.sigmoid(model(x, t if use_tau else None)).cpu())
    return torch.cat(out).numpy()

def pair_auc(model, D, idx, fam, use_tau=False):
    it = task_items(idx, fam)
    y = [0] * len(idx) + [1] * len(idx)
    return auc(y, predict(model, D, it, use_tau))

@torch.no_grad()
def freq_anom(model, D, items, eps=0.05, bs=64):
    """A(x): mean over low/mid/high Fourier bands of ||f(x+d)-f(x)||/||d|| (Eq. 12)."""
    model.eval(); r = D.radial.r; S = D.S
    bands = [(0, .15), (.15, .4), (.4, 1.01)]
    g = torch.Generator().manual_seed(0)
    out = []
    for k in range(0, len(items), bs):
        x, _, _ = D.batch(items[k:k + bs])
        f0 = model.feat(x)
        acc = torch.zeros(len(x), device=DEV)
        for lo, hi in bands:
            n = torch.randn(len(x), S, S, generator=g)
            Fn = torch.fft.fftshift(torch.fft.fft2(n), dim=(-2, -1)) * ((r >= lo) & (r < hi)).float()
            d = torch.fft.ifft2(torch.fft.ifftshift(Fn, dim=(-2, -1))).real
            d = d / d.flatten(1).norm(dim=1).view(-1, 1, 1) * eps * S  # fixed L2 budget
            d = d.unsqueeze(1).expand(-1, 3, -1, -1).to(DEV)
            f1 = model.feat((x + d).clamp(0, 1))
            acc += (f1 - f0).norm(dim=1) / d.flatten(1).norm(dim=1)
        out.append((acc / len(bands)).cpu())
    return torch.cat(out).numpy()

# ---------------------------------------------------------------- CLIP (zero-shot head + UnivFD-style probe)
def clip_features(D, cache):
    if os.path.exists(cache):
        return np.load(cache)
    from transformers import CLIPModel
    m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEV).eval()
    mean = torch.tensor([0.4815, 0.4578, 0.4082]).view(1, 3, 1, 1).to(DEV)
    std = torch.tensor([0.2686, 0.2613, 0.2758]).view(1, 3, 1, 1).to(DEV)
    feats = np.zeros((D.N, 4, 512), np.float32)
    with torch.no_grad():
        for s in range(4):
            for k in range(0, D.N, 128):
                x = D.x[k:k + 128, s].float().div(255).to(DEV)
                x = F.interpolate(x, size=224, mode="bicubic", align_corners=False).clamp(0, 1)
                e = m.visual_projection(m.vision_model(pixel_values=(x - mean) / std).pooler_output)
                feats[k:k + 128, s] = F.normalize(e, dim=-1).cpu().numpy()
    np.save(cache, feats)
    return feats

def clip_text(prompts):
    from transformers import CLIPModel, CLIPTokenizer
    m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    with torch.no_grad():
        t = tok(prompts, padding=True, return_tensors="pt")
        e = m.text_projection(m.text_model(**t).pooler_output)
    return F.normalize(e, dim=-1).numpy()

# ---------------------------------------------------------------- federated training pieces
def local_train(model, D, items, cfg, hp, gvec=None, fisher=None, anchor=None, mem=None, use_tau=False):
    model.train()
    ps = params(model)
    opt = torch.optim.AdamW(ps, lr=hp["lr"], weight_decay=0.0)
    for _ in range(hp["local_epochs"]):
        order = np.random.permutation(len(items))
        for k in range(0, len(order), hp["bs"]):
            sel = [items[j] for j in order[k:k + hp["bs"]]]
            if len(sel) < 4:
                continue
            x, y, t = D.batch(sel)
            loss = F.binary_cross_entropy_with_logits(model(x, t if use_tau else None), y)
            if cfg.get("mu") and gvec is not None:
                loss = loss + cfg["mu"] / 2 * ((flat(ps) - gvec) ** 2).sum()
            if fisher is not None:
                loss = loss + hp["lam"] / 2 * (fisher * (flat(ps) - anchor) ** 2).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if mem is not None:  # GEM: project against the episodic-memory gradient
                g = flat([p.grad if p.grad is not None else torch.zeros_like(p) for p in ps])
                mb = [mem[j] for j in np.random.choice(len(mem), min(32, len(mem)), replace=False)]
                xm, ym, tm = D.batch(mb)
                opt.zero_grad(set_to_none=True)
                F.binary_cross_entropy_with_logits(model(xm, tm if use_tau else None), ym).backward()
                gm = flat([p.grad if p.grad is not None else torch.zeros_like(p) for p in ps])
                dot = (g * gm).sum()
                if dot < 0:
                    g = g - dot / (gm * gm).sum().clamp(min=1e-12) * gm
                i = 0
                for p in ps:
                    n = p.numel(); p.grad = g[i:i + n].view_as(p).clone(); i += n
            opt.step()

def fisher_diag(model, D, items, use_tau, nb=24, bs=8):
    model.eval(); ps = params(model); acc = torch.zeros(sum(p.numel() for p in ps), device=DEV)
    for _ in range(nb):
        sel = [items[j] for j in np.random.choice(len(items), min(bs, len(items)), replace=False)]
        x, y, t = D.batch(sel)
        model.zero_grad()
        F.binary_cross_entropy_with_logits(model(x, t if use_tau else None), y).backward()
        acc += flat([p.grad if p.grad is not None else torch.zeros_like(p) for p in ps]) ** 2
    return acc / nb

def quantize(delta, mode):
    """Return (dequantised delta, uplink bytes) for one parameter tensor."""
    n = delta.numel()
    if mode is None:
        return delta, 4 * n
    if mode == "int8":
        sc = delta.abs().max().clamp(min=1e-12) / 127
        return torch.round(delta / sc).clamp(-127, 127) * sc, n + 4
    k = max(1, int(0.3 * n))  # sparse: top-30% by magnitude, int8 values + 1-bit mask
    thr = delta.abs().flatten().kthvalue(n - k + 1).values
    keep = delta.abs() >= thr
    sc = delta.abs().max().clamp(min=1e-12) / 127
    return torch.where(keep, torch.round(delta / sc).clamp(-127, 127) * sc, torch.zeros_like(delta)), k + math.ceil(n / 8) + 4

def state_bytes_buffers(sd):
    return sum(4 * v.numel() for k, v in sd.items() if v.dtype.is_floating_point and ("running" in k))

def new_model(D):
    return Net(D.radial).to(DEV)

def average_into(gm, deltas, weights, bufs):
    """Eq. 1 on parameters; weighted average of BN running statistics."""
    with torch.no_grad():
        for name, p in gm.named_parameters():
            p.add_(sum(w * d[name] for w, d in zip(weights, deltas)))
        for name, b in gm.named_buffers():
            if b.dtype.is_floating_point and "running" in name:
                b.copy_(sum(w * bf[name] for w, bf in zip(weights, bufs)))

def run_fed(D, cfg, hp, known, seed, log):
    alpha = hp.get('alpha', 0.5)
    torch.manual_seed(seed); np.random.seed(seed)
    use_tau = cfg.get("prov", False)
    parts = partition(D.train, hp["clients"], seed, alpha)
    nC = len(parts)
    gm = new_model(D); loc = new_model(D)
    R = hp["rounds"]; boundary = R // 2
    proxy_items = [(int(i), s) for i in D.proxy for s in (0, known[0], known[1])]
    res = dict(val_curve=[], uplink_mb=[], round_time=[])
    fisher_G = None; anchor = None; fishers = None; mems = None
    val_idx = D.val
    for r in range(R):
        t0 = time.time()
        task = 0 if r < boundary else 1
        fam = known[task]
        if r == boundary:
            res["auc_t1_mid"] = pair_auc(gm, D, D.test, known[0], use_tau)
            if cfg.get("ewc"):
                anchor = flat(params(gm)).detach().clone()
                fishers = []
                for k in range(nC):
                    loc.load_state_dict(gm.state_dict())
                    fishers.append(fisher_diag(loc, D, task_items(parts[k], known[0]), use_tau))
                w = torch.tensor([len(p) for p in parts], dtype=torch.float, device=DEV); w /= w.sum()
                fisher_G = sum(wi * f for wi, f in zip(w, fishers))
            if cfg.get("gem"):
                mems = [task_items(parts[k][:max(1, min(64, len(parts[k])))], known[0]) for k in range(nC)]
        sel = np.random.choice(nC, min(hp["K"], nC), replace=False)
        gvec = flat(params(gm)).detach().clone()
        gstate = {k: v.clone() for k, v in gm.state_dict().items()}
        deltas, bufs, sizes, up = [], [], [], 0
        cstates = []
        for k in sel:
            loc.load_state_dict(gstate)
            fk = None; anc = None; mem = None
            if task == 1 and cfg.get("ewc"):
                fk = fisher_G if cfg["ewc"] == "fed" else fishers[k]; anc = anchor
            if task == 1 and cfg.get("gem"):
                mem = mems[k]
            local_train(loc, D, task_items(parts[k], fam), cfg, hp, gvec if cfg.get("mu") else None, fk, anc, mem, use_tau)
            d = {}
            for name, p in loc.named_parameters():
                dq, nb = quantize((p.detach() - dict(gm.named_parameters())[name].detach()), cfg.get("quant"))
                d[name] = dq; up += nb
            up += state_bytes_buffers(loc.state_dict())
            deltas.append(d); sizes.append(len(parts[k]))
            bufs.append({n: b.detach().clone() for n, b in loc.named_buffers() if "running" in n})
            if cfg.get("kd"):
                cstates.append({k2: v.detach().clone() for k2, v in loc.state_dict().items()})
        w = np.array(sizes, float); w /= w.sum()
        average_into(gm, deltas, w, bufs)
        if cfg.get("kd"):  # Eq. 2: distil the client ensemble into the global model on the proxy set P
            ens = []
            for st in cstates:
                loc.load_state_dict(st)
                ens.append(predict(loc, D, proxy_items, use_tau))
            tgt = torch.tensor(np.mean(ens, 0), dtype=torch.float, device=DEV)
            gm.train(); opt = torch.optim.AdamW(params(gm), lr=hp["lr"])
            for _ in range(hp["kd_steps"]):
                j = np.random.choice(len(proxy_items), hp["bs"], replace=False)
                x, _, t = D.batch([proxy_items[q] for q in j])
                l = hp["lam_kd"] * F.binary_cross_entropy_with_logits(gm(x, t if use_tau else None), tgt[j])
                opt.zero_grad(); l.backward(); opt.step()
        # per-round validation AUC: real vs both known fakes
        va = np.mean([pair_auc(gm, D, val_idx, known[0], use_tau), pair_auc(gm, D, val_idx, known[1], use_tau)])
        res["val_curve"].append(va); res["uplink_mb"].append(up / len(sel) / 1e6)
        res["round_time"].append(time.time() - t0)
        log(f"  r{r+1:02d}/{R} task{task+1} val_auc={va:.4f} uplink/client={up/len(sel)/1e6:.1f}MB {time.time()-t0:.0f}s")
    return gm, res

def run_central(D, hp, known, seed, log):
    torch.manual_seed(seed); np.random.seed(seed)
    m = new_model(D)
    items = [(int(i), s) for i in D.train for s in (0, known[0], known[1])]
    h = dict(hp); h["local_epochs"] = hp["central_epochs"]
    local_train(m, D, items, {}, h)
    return m, dict(val_curve=[])

# ---------------------------------------------------------------- open-set evaluation
def open_set(model, D, clipf, txt, known, held, use_tau, seed):
    test = D.test
    items_real = [(int(i), 0) for i in test]
    items_unseen = [(int(i), held) for i in test]
    res = {}
    def tv(items): return torch.tensor([D.tau[it] for it in items])
    it_all = items_real + items_unseen
    y = [0] * len(test) + [1] * len(test)
    p = predict(model, D, it_all, use_tau)
    res["ckm_only"] = auc(y, p)
    # CLIP zero-shot: p_real / p_fake prompts, tau=0.07 as in Eq. 11 (cosine / tau)
    ci = np.concatenate([clipf[test, 0], clipf[test, held]])
    lg = ci @ txt.T / 0.07
    s_vlm = np.exp(lg[:, 1]) / (np.exp(lg[:, 0]) + np.exp(lg[:, 1]))
    res["clip_zeroshot"] = auc(y, s_vlm)
    # UnivFD-style: linear probe on frozen CLIP features, trained on known families only
    Xtr = np.concatenate([clipf[D.train, 0], clipf[D.train, known[0]], clipf[D.train, known[1]]])
    ytr = [0] * len(D.train) + [1] * (2 * len(D.train))
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
    res["univfd_style"] = auc(y, lr.decision_function(ci))
    # FreqAnom, z-scored on validation reals (no unseen data used for calibration)
    A = freq_anom(model, D, it_all)
    Av = freq_anom(model, D, [(int(i), 0) for i in D.val])
    z = (A - Av.mean()) / (Av.std() + 1e-8)
    sg = 1 / (1 + np.exp(-z))
    res["freqanom_only"] = auc(y, sg)
    al, be, ga = 0.7, 0.15, 0.15  # fixed a priori, not tuned on the unseen family
    res["fusion_clip"] = auc(y, al * p + (1 - al) * s_vlm)
    res["fusion_freq"] = auc(y, al * p + (1 - al) * sg)
    res["fusion_full"] = auc(y, al * p + be * s_vlm + ga * sg)
    if use_tau:  # provenance: tau=0 raises gamma by 0.1 (taken from alpha), Sec. III-F
        t = tv(it_all).numpy()
        a2 = np.where(t == 0, al - 0.1, al); g2 = np.where(t == 0, ga + 0.1, ga)
        res["fusion_full_prov"] = auc(y, a2 * p + be * s_vlm + g2 * sg)
    # closed-set detection on the two known families and unseen recall at 5% FPR on real val
    thr = np.quantile(predict(model, D, [(int(i), 0) for i in D.val], use_tau), 0.95)
    res["unseen_recall_at_5fpr_ckm"] = float((p[len(test):] > thr).mean())
    return res

def tau_fpr(model, D, use_tau):
    """False-positive rate on authentic test images per tau level, at the 0.5 threshold."""
    items = [(int(i), 0) for i in D.test]
    p = predict(model, D, items, use_tau)
    t = np.array([D.tau[it] for it in items])
    return {str(l): (float((p[t == l] > 0.5).mean()) if (t == l).sum() else None) for l in (0.0, 0.3, 0.7, 1.0)} | \
           {"n": {str(l): int((t == l).sum()) for l in (0.0, 0.3, 0.7, 1.0)}}

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])  # held-out generator index
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--central_epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lam", type=float, default=1000.0)
    ap.add_argument("--lam_kd", type=float, default=0.4)
    ap.add_argument("--kd_steps", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration for client quantity skew")
    ap.add_argument("--tag", default="", help="suffix for result files of sensitivity runs")
    ap.add_argument("--save_models", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    hp = dict(rounds=a.rounds, clients=a.clients, K=a.K, local_epochs=a.local_epochs, central_epochs=a.central_epochs,
              bs=a.bs, lr=a.lr, lam=a.lam, lam_kd=a.lam_kd, kd_steps=a.kd_steps, alpha=a.alpha)
    logf = open(os.path.join(a.out, "log.txt"), "a")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    log(f"device={DEV} hp={hp}")
    for seed in a.seeds:
        D = Data(a.data, seed)
        clipf = clip_features(D, a.data.replace(".npz", "_clip.npy"))
        txt = clip_text(["an authentic unmanipulated human face photo", "an AI-generated or digitally manipulated fake face photo"])
        for held in a.folds:
            known = [s for s in (1, 2, 3) if s != held]  # task order: lower index first
            for name in a.methods:
                fn = os.path.join(a.out, f"s{seed}_held{held}_{name}" + (f"__{a.tag}" if a.tag else "") + ".json")
                if os.path.exists(fn):
                    log(f"skip {fn}"); continue
                cfg = METHODS[name]; t0 = time.time()
                log(f"== seed{seed} held={SRC[held]} known={[SRC[k] for k in known]} method={name}")
                if name == "central":
                    m, res = run_central(D, hp, known, seed, log)
                    res["auc_t1_end"] = pair_auc(m, D, D.test, known[0]); res["auc_t2_end"] = pair_auc(m, D, D.test, known[1])
                    res["auc_t1_mid"] = res["auc_t1_end"]; use_tau = False
                else:
                    m, res = run_fed(D, cfg, hp, known, seed, log)
                    use_tau = cfg.get("prov", False)
                    res["auc_t1_end"] = pair_auc(m, D, D.test, known[0], use_tau)
                    res["auc_t2_end"] = pair_auc(m, D, D.test, known[1], use_tau)
                res["bwt"] = res["auc_t1_end"] - res["auc_t1_mid"]
                res["open_set"] = open_set(m, D, clipf, txt, known, held, use_tau, seed)
                res["tau_fpr"] = tau_fpr(m, D, use_tau)
                res.update(method=name, tag=a.tag, seed=seed, held=SRC[held], known=[SRC[k] for k in known], hp=hp, minutes=(time.time() - t0) / 60)
                json.dump(res, open(fn, "w"), indent=1)
                if a.save_models and name in ("full", "kd_ewc_gem"):
                    torch.save(m.state_dict(), os.path.join(a.out, f"s{seed}_held{held}_{name}.pt"))
                log(f"   t1_mid={res['auc_t1_mid']:.4f} t1_end={res['auc_t1_end']:.4f} t2_end={res['auc_t2_end']:.4f} "
                    f"bwt={res['bwt']:+.4f} unseen_ckm={res['open_set']['ckm_only']:.4f} ({res['minutes']:.1f} min)")

if __name__ == "__main__":
    main()
