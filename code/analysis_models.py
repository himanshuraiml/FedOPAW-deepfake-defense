"""Model-based diagnostics on the saved full-run checkpoints (seed 0): spectra, ROC curves, score
distributions, per-band sensitivity, and a feature-space projection. Output: results_full/analysis.npz
Usage: python analysis_models.py --data data/dff_n6000_s128.npz --results results_full
"""
import argparse, os, pickle
import numpy as np, torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import fedopaw_pilot as P

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--results", default="results_full")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n", type=int, default=600)
a = ap.parse_args()
D = P.Data(a.data, a.seed)
clipf = P.clip_features(D, a.data.replace(".npz", "_clip.npy"))
txt = P.clip_text(["an authentic unmanipulated human face photo", "an AI-generated or digitally manipulated fake face photo"])
out = {"sources": P.SRC}

# 1. data-level radial spectra (test identities), no model involved
test = D.test[: a.n]
out["spectrum"] = np.stack([D.radial.raw(D.x[test, s].float() / 255).mean(0).numpy() for s in range(4)])
out["spectrum_std"] = np.stack([D.radial.raw(D.x[test, s].float() / 255).std(0).numpy() for s in range(4)])

@torch.no_grad()
def band_sens(model, items, eps=0.05, bs=64):
    model.eval(); r = D.radial.r; S = D.S; g = torch.Generator().manual_seed(0)
    bands = [(0, .15), (.15, .4), (.4, 1.01)]; res = []
    for k in range(0, len(items), bs):
        x, _, _ = D.batch(items[k:k + bs]); f0 = model.feat(x); row = []
        for lo, hi in bands:
            n = torch.randn(len(x), S, S, generator=g)
            Fn = torch.fft.fftshift(torch.fft.fft2(n), dim=(-2, -1)) * ((r >= lo) & (r < hi)).float()
            d = torch.fft.ifft2(torch.fft.ifftshift(Fn, dim=(-2, -1))).real
            d = d / d.flatten(1).norm(dim=1).view(-1, 1, 1) * eps * S
            d = d.unsqueeze(1).expand(-1, 3, -1, -1).to(P.DEV)
            row.append(((model.feat((x + d).clamp(0, 1)) - f0).norm(dim=1) / d.flatten(1).norm(dim=1)).cpu())
        res.append(torch.stack(row, 1))
    return torch.cat(res).numpy()

@torch.no_grad()
def feats(model, items, bs=128):
    model.eval(); o = []
    for k in range(0, len(items), bs):
        x, _, _ = D.batch(items[k:k + bs]); o.append(model.feat(x).cpu())
    return torch.cat(o).numpy()

for held in (1, 2, 3):
    known = [s for s in (1, 2, 3) if s != held]
    m = P.new_model(D); m.load_state_dict(torch.load(os.path.join(a.results, f"s{a.seed}_held{held}_kd_ewc_gem.pt"), map_location=P.DEV)); m.eval()
    ti = D.test[: a.n]
    it_r = [(int(i), 0) for i in ti]; it_u = [(int(i), held) for i in ti]; it_k = [(int(i), known[0]) for i in ti] + [(int(i), known[1]) for i in ti]
    y = np.array([0] * len(ti) + [1] * len(ti))
    p = P.predict(m, D, it_r + it_u)
    lg = np.concatenate([clipf[ti, 0], clipf[ti, held]]) @ txt.T / 0.07
    s_vlm = np.exp(lg[:, 1]) / (np.exp(lg[:, 0]) + np.exp(lg[:, 1]))
    Xtr = np.concatenate([clipf[D.train, 0], clipf[D.train, known[0]], clipf[D.train, known[1]]]); ytr = [0] * len(D.train) + [1] * (2 * len(D.train))
    probe = LogisticRegression(max_iter=2000).fit(Xtr, ytr).decision_function(np.concatenate([clipf[ti, 0], clipf[ti, held]]))
    A = P.freq_anom(m, D, it_r + it_u); Av = P.freq_anom(m, D, [(int(i), 0) for i in D.val])
    sg = 1 / (1 + np.exp(-(A - Av.mean()) / (Av.std() + 1e-8)))
    fus = 0.7 * p + 0.15 * s_vlm + 0.15 * sg
    sc = {"Detector": p, "CLIP zero-shot": s_vlm, "CLIP probe": probe, "FreqAnom": sg, "Fusion": fus}
    out[f"roc_held{held}"] = {k: roc_curve(y, v)[:2] for k, v in sc.items()}
    out[f"auc_held{held}"] = {k: float(roc_auc_score(y, v)) for k, v in sc.items()}
    # A(x) by band: real vs known fakes vs unseen
    out[f"band_held{held}"] = {"real": band_sens(m, it_r), "known": band_sens(m, it_k[: len(ti)]), "unseen": band_sens(m, it_u)}
    out[f"A_held{held}"] = {"real": A[: len(ti)], "unseen": A[len(ti):], "val_real": Av}
    out[f"score_held{held}"] = {"real": p[: len(ti)], "unseen": p[len(ti):], "known": P.predict(m, D, it_k)}
    if held == 1:
        F_ = np.concatenate([feats(m, it_r), feats(m, it_k[: len(ti)]), feats(m, it_k[len(ti):]), feats(m, it_u)])
        Z = PCA(2, random_state=0).fit_transform(F_)
        lab = np.array([0] * len(ti) + [known[0]] * len(ti) + [known[1]] * len(ti) + [held] * len(ti))
        out["pca"] = (Z, lab)
    print("fold", held, {k: round(v, 3) for k, v in out[f"auc_held{held}"].items()})
pickle.dump(out, open(os.path.join(a.results, "analysis.pkl"), "wb"))
print("saved")
