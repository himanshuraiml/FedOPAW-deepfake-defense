"""Extra figures, statistics and sensitivity tables from results_full (+ analysis.pkl, results_sweep).
Everything is computed from the run outputs. Usage: python extras.py --results results_full --sweep results_sweep --out generated_full
"""
import argparse, glob, json, os, pickle
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results_full"); ap.add_argument("--sweep", default="results_sweep"); ap.add_argument("--out", default="generated_full")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
C = plt.get_cmap("tab10")

R = {}
for f in sorted(glob.glob(os.path.join(a.results, "s*_held*_*.json"))):
    if "__" in os.path.basename(f): continue
    r = json.load(open(f)); R.setdefault(r["method"], {})[(r["seed"], r["held"])] = r
WM = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.results, "wm_s*_held*.json")))]
HELD = ["inpainting", "insight", "text2img"]
def closed(r): return (r["auc_t1_end"] + r["auc_t2_end"]) / 2
M = {}  # macros

def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(a.out, name), dpi=220); plt.close(fig)

# ---------------------------------------------------------------- statistics table (paired over seed x fold)
def paired(m1, m2, get):
    keys = sorted(set(R[m1]) & set(R[m2]))
    x = np.array([get(R[m1][k]) for k in keys]); y = np.array([get(R[m2][k]) for k in keys])
    d = x - y; n = len(d)
    ci = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=stats.sem(d)) if d.std() > 0 else (d.mean(), d.mean())
    p = stats.wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0
    return d.mean(), ci, p, n
ck = closed
def os_(key): return lambda r: r["open_set"][key]
rows = [("FedAvg vs centralized (closed AUC)", "fedavg", "central", ck, 100),
        ("FedOPAW vs FedAvg (closed AUC)", "full", "fedavg", ck, 100),
        ("+Federated KD vs FedAvg (closed AUC)", "kd", "fedavg", ck, 100),
        ("+Fed-EWC vs +KD (closed AUC)", "kd_ewc", "kd", ck, 100),
        ("+GEM vs +Fed-EWC (closed AUC)", "kd_ewc_gem", "kd_ewc", ck, 100),
        ("FedAvg+EWC vs FedAvg (BWT, AUC pts)", "fedavg_ewc", "fedavg", lambda r: r["bwt"], 100),
        ("+Fed-EWC vs +KD (BWT, AUC pts)", "kd_ewc", "kd", lambda r: r["bwt"], 100),
        ("+GEM vs +Fed-EWC (BWT, AUC pts)", "kd_ewc_gem", "kd_ewc", lambda r: r["bwt"], 100),
        ("Detector: FedOPAW-stack vs FedAvg (unseen AUROC)", "kd_ewc_gem", "fedavg", os_("ckm_only"), 100),
        ("Fusion vs detector alone (unseen AUROC)", "kd_ewc_gem", "kd_ewc_gem", None, 100),
        ("+CLIP vs detector alone (unseen AUROC)", "kd_ewc_gem", "kd_ewc_gem", None, 100),
        ("+Provenance vs fusion (unseen AUROC)", "full", "kd_ewc_gem", None, 100),
        ("Detector vs CLIP probe (unseen AUROC)", "kd_ewc_gem", "fedavg", None, 100)]
s = "\\begin{tabular}{@{}lccc@{}}\n\\toprule\nComparison (A vs B) & Mean $\\Delta$ & 95\\% CI & Wilcoxon $p$ \\\\\n\\midrule\n"
sig = []
for name, m1, m2, get, k in rows:
    if get is None:  # comparisons inside or across methods on different keys
        if name.startswith("Fusion vs"): x = [R["kd_ewc_gem"][q]["open_set"]["fusion_full"] - R["kd_ewc_gem"][q]["open_set"]["ckm_only"] for q in R["kd_ewc_gem"]]
        elif name.startswith("+CLIP"): x = [R["kd_ewc_gem"][q]["open_set"]["fusion_clip"] - R["kd_ewc_gem"][q]["open_set"]["ckm_only"] for q in R["kd_ewc_gem"]]
        elif name.startswith("+Provenance"): x = [R["full"][q]["open_set"]["fusion_full_prov"] - R["kd_ewc_gem"][q]["open_set"]["fusion_full"] for q in R["kd_ewc_gem"]]
        else: x = [R["kd_ewc_gem"][q]["open_set"]["ckm_only"] - R["fedavg"][q]["open_set"]["univfd_style"] for q in R["kd_ewc_gem"]]
        x = np.array(x); n = len(x)
        ci = stats.t.interval(0.95, n - 1, loc=x.mean(), scale=stats.sem(x)); p = stats.wilcoxon(x).pvalue; mean = x.mean()
    else:
        mean, ci, p, n = paired(m1, m2, get)
    s += f"{name} & {mean*k:+.2f} & [{ci[0]*k:+.2f}, {ci[1]*k:+.2f}] & {p:.3f} \\\\\n"
    sig.append((name, mean * k, p))
s += "\\bottomrule\n\\end{tabular}\n"
open(os.path.join(a.out, "tab_stats.tex"), "w").write(s)
M["NPairs"] = str(n)

# ---------------------------------------------------------------- per-fold table
s = "\\begin{tabular}{@{}llccc@{}}\n\\toprule\nMetric & Method & Inpaint. & InsightF. & Text2Img \\\\\n\\midrule\n"
def cell(m, held, get):
    v = [get(r) for (sd, h), r in R[m].items() if h == held]
    return f"{100*np.mean(v):.1f}"
for met, get_fn, meths in [("Closed AUC", ck, [("Centralized", "central"), ("FedAvg", "fedavg"), ("FedOPAW", "full")]),
                           ("Unseen AUROC", os_("ckm_only"), [("FedAvg", "fedavg"), ("FedOPAW stack", "kd_ewc_gem")])]:
    for i, (nm, m) in enumerate(meths):
        s += (f"{met} & " if i == 0 else "& ") + f"{nm} & " + " & ".join(cell(m, h, get_fn) for h in HELD) + " \\\\\n"
    s += "\\midrule\n"
s += "Unseen AUROC & CLIP probe & " + " & ".join(cell("fedavg", h, os_("univfd_style")) for h in HELD) + " \\\\\n"
s += "Recall@5\\%FPR & FedOPAW stack & " + " & ".join(cell("kd_ewc_gem", h, os_("unseen_recall_at_5fpr_ckm")) for h in HELD) + " \\\\\n"
s += "\\bottomrule\n\\end{tabular}\n"
open(os.path.join(a.out, "tab_perfold.tex"), "w").write(s)
M["RecallFive"] = f"{100*np.mean([r['open_set']['unseen_recall_at_5fpr_ckm'] for r in R['kd_ewc_gem'].values()]):.1f}"
M["RecallFiveAvg"] = f"{100*np.mean([r['open_set']['unseen_recall_at_5fpr_ckm'] for r in R['fedavg'].values()]):.1f}"

# ---------------------------------------------------------------- Fig: closed AUC per task
meths = [("Central", "central"), ("FedAvg", "fedavg"), ("FedProx", "fedprox"), ("FedAvg+EWC", "fedavg_ewc"), ("FedOPAW", "full")]
fig, ax = plt.subplots(figsize=(3.4, 2.3)); w = 0.38
for j, (key, lab) in enumerate([("auc_t1_end", "Task 1 (learned first)"), ("auc_t2_end", "Task 2 (learned last)")]):
    mu = [100 * np.mean([r[key] for r in R[m].values()]) for _, m in meths]; sd = [100 * np.std([r[key] for r in R[m].values()], ddof=1) for _, m in meths]
    ax.bar(np.arange(len(meths)) + (j - .5) * w, mu, w, yerr=sd, capsize=2, label=lab, color=C(j))
ax.set_xticks(range(len(meths))); ax.set_xticklabels([n for n, _ in meths]); ax.set_ylim(94, 100); ax.set_ylabel("Test AUC (%)"); ax.legend(loc="lower left")
save(fig, "fig_closed.png")

# ---------------------------------------------------------------- Fig: BWT strip plot
fig, ax = plt.subplots(figsize=(3.4, 2.3))
order = [("FedAvg", "fedavg"), ("FedProx", "fedprox"), ("FedAvg\n+EWC", "fedavg_ewc"), ("+KD", "kd"), ("+Fed-\nEWC", "kd_ewc"), ("+GEM", "kd_ewc_gem"), ("FedOPAW", "full")]
rng = np.random.RandomState(0)
for i, (lab, m) in enumerate(order):
    for (sd, h), r in R[m].items():
        ax.scatter(i + rng.uniform(-.18, .18), 100 * r["bwt"], s=9, color=C(HELD.index(r["held"])), alpha=.85)
    ax.hlines(100 * np.mean([r["bwt"] for r in R[m].values()]), i - .3, i + .3, color="k", lw=1.2)
ax.axhline(0, color="gray", lw=.6, ls=":"); ax.set_xticks(range(len(order))); ax.set_xticklabels([l for l, _ in order])
ax.set_ylabel("BWT (AUC points)")
for j, h in enumerate(HELD): ax.scatter([], [], s=9, color=C(j), label=f"hold out {h}")
ax.legend(loc="upper right", ncol=1); save(fig, "fig_bwt.png")

# ---------------------------------------------------------------- Fig: ablation (2 panels)
steps = [("FedAvg", "fedavg", "ckm_only"), ("+KD", "kd", "ckm_only"), ("+Fed-EWC", "kd_ewc", "ckm_only"), ("+GEM", "kd_ewc_gem", "ckm_only"),
         ("+CLIP", "kd_ewc_gem", "fusion_clip"), ("+Freq", "kd_ewc_gem", "fusion_freq"), ("+Fusion", "kd_ewc_gem", "fusion_full"), ("+Prov", "full", "fusion_full_prov")]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.2))
for ax, (ttl, get) in zip(axs, [("Closed-set AUC (%)", lambda m, k, r: closed(r)), ("Unseen-generator AUROC (%)", lambda m, k, r: r["open_set"][k])]):
    mu = [100 * np.mean([get(m, k, r) for r in R[m].values()]) for _, m, k in steps]
    sd = [100 * np.std([get(m, k, r) for r in R[m].values()], ddof=1) for _, m, k in steps]
    ax.errorbar(range(len(steps)), mu, yerr=sd, fmt="o-", capsize=2, ms=3, color=C(0)); ax.set_xticks(range(len(steps))); ax.set_xticklabels([n for n, _, _ in steps])
    ax.set_title(ttl); ax.grid(alpha=.25)
save(fig, "fig_ablation.png")

# ---------------------------------------------------------------- Fig: heatmaps per held-out generator
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
rows_ = [("Centralized", "central", ck), ("FedAvg", "fedavg", ck), ("FedProx", "fedprox", ck), ("FedAvg+EWC", "fedavg_ewc", ck), ("FedOPAW", "full", ck)]
H = np.array([[100 * np.mean([g(r) for (sd, h), r in R[m].items() if h == hh]) for hh in HELD] for _, m, g in rows_])
im = axs[0].imshow(H, cmap="viridis", aspect="auto"); axs[0].set_title("Closed-set AUC (%) by held-out generator")
axs[0].set_yticks(range(len(rows_))); axs[0].set_yticklabels([n for n, _, _ in rows_]); axs[0].set_xticks(range(3)); axs[0].set_xticklabels(HELD)
for i in range(H.shape[0]):
    for j in range(3): axs[0].text(j, i, f"{H[i,j]:.1f}", ha="center", va="center", color="w", fontsize=6)
rows2 = [("CLIP zero-shot", "fedavg", "clip_zeroshot"), ("CLIP probe", "fedavg", "univfd_style"), ("FreqAnom", "kd_ewc_gem", "freqanom_only"), ("Detector", "kd_ewc_gem", "ckm_only"),
         ("Fusion", "kd_ewc_gem", "fusion_full"), ("Fusion+prov", "full", "fusion_full_prov")]
H2 = np.array([[100 * np.mean([r["open_set"][k] for (sd, h), r in R[m].items() if h == hh]) for hh in HELD] for _, m, k in rows2])
axs[1].imshow(H2, cmap="viridis", aspect="auto", vmin=40, vmax=100); axs[1].set_title("Unseen-generator AUROC (%)")
axs[1].set_yticks(range(len(rows2))); axs[1].set_yticklabels([n for n, _, _ in rows2]); axs[1].set_xticks(range(3)); axs[1].set_xticklabels(["hold out\n" + h for h in HELD])
for i in range(H2.shape[0]):
    for j in range(3): axs[1].text(j, i, f"{H2[i,j]:.1f}", ha="center", va="center", color="w", fontsize=6)
save(fig, "fig_heat.png")

# ---------------------------------------------------------------- Fig: provenance FPR and communication
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.2))
lv = ["0.0", "0.3", "1.0"]; w = 0.38
for j, (lab, m) in enumerate([("without $\\tau$", "kd_ewc_gem"), ("with $\\tau$", "full")]):
    mu = [100 * np.mean([r["tau_fpr"][l] for r in R[m].values() if r["tau_fpr"][l] is not None]) for l in lv]
    sd = [100 * np.std([r["tau_fpr"][l] for r in R[m].values() if r["tau_fpr"][l] is not None], ddof=1) for l in lv]
    axs[0].bar(np.arange(3) + (j - .5) * w, mu, w, yerr=sd, capsize=2, label=lab, color=C(j))
axs[0].set_xticks(range(3)); axs[0].set_xticklabels(["$\\tau$=0 (no manifest)", "$\\tau$=0.3", "$\\tau$=1.0 (valid)"]); axs[0].set_ylabel("FPR on authentic images (%)"); axs[0].legend()
for name, m, c in [("FedAvg fp32", "fedavg", 0), ("int8", "full", 1), ("int8 + 30% sparse", "full_sparse", 2)]:
    up = np.mean([np.mean(r["uplink_mb"]) for r in R[m].values()]); au = [100 * closed(r) for r in R[m].values()]
    axs[1].errorbar(up, np.mean(au), yerr=np.std(au, ddof=1), fmt="o", color=C(c), capsize=2, label=name)
axs[1].set_xscale("log"); axs[1].set_xlabel("Uplink per client per round (MB, log scale)"); axs[1].set_ylabel("Closed-set AUC (%)"); axs[1].legend(); axs[1].grid(alpha=.25)
save(fig, "fig_prov_comm.png")

# ---------------------------------------------------------------- Fig: watermark
if WM:
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3)); kinds = ["clean", "gauss", "jpeg", "crop"]; klab = ["Clean", "Gaussian", "JPEG Q50", "20% crop"]
    vars_ = [("naive", "naive mark"), ("det_ft", "detector-only"), ("emb_only", "embedder-only"), ("advmark", "AdvMark-Fed")]; w = 0.2
    for j, (v, lab) in enumerate(vars_):
        mu = [np.mean([x["rows"][k][v]["delta_auc"] for x in WM]) for k in kinds]; sd = [np.std([x["rows"][k][v]["delta_auc"] for x in WM], ddof=1) for k in kinds]
        axs[0].bar(np.arange(4) + (j - 1.5) * w, mu, w, yerr=sd, capsize=1.5, label=lab, color=C(j))
    axs[0].axhline(0, color="k", lw=.6); axs[0].set_xticks(range(4)); axs[0].set_xticklabels(klab); axs[0].set_ylabel("$\\Delta$AUC (points)"); axs[0].legend(ncol=2)
    for j, (v, lab) in enumerate([("naive", "naive"), ("advmark", "AdvMark-Fed")]):
        mu = [np.mean([x["rows"][k][v]["wba_authentic"] for x in WM]) for k in kinds]
        axs[1].bar(np.arange(4) + (j - .5) * .38, mu, .38, label=lab, color=C(4 + j))
    axs[1].axhline(50, color="gray", ls=":", lw=.7); axs[1].set_xticks(range(4)); axs[1].set_xticklabels(klab); axs[1].set_ylabel("Watermark bit accuracy (%)"); axs[1].set_ylim(40, 102); axs[1].legend()
    save(fig, "fig_wm.png")

# ---------------------------------------------------------------- model-based figures
pk = os.path.join(a.results, "analysis.pkl")
if os.path.exists(pk):
    A = pickle.load(open(pk, "rb")); src = ["real", "inpainting", "insight", "text2img"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.2)); x = np.arange(32)
    for s_ in range(4):
        axs[0].plot(x, A["spectrum"][s_], label=src[s_], color=C(s_))
        if s_: axs[1].plot(x, A["spectrum"][s_] - A["spectrum"][0], label=src[s_], color=C(s_))
    axs[0].set_xlabel("Radial frequency bin (0 = DC, 31 = Nyquist)"); axs[0].set_ylabel("Mean log power"); axs[0].legend()
    axs[1].axhline(0, color="k", lw=.6); axs[1].set_xlabel("Radial frequency bin"); axs[1].set_ylabel("Log power minus real"); axs[1].legend()
    save(fig, "fig_spectrum.png")
    d = (A["spectrum"][1:] - A["spectrum"][0]); M["SpecHighInp"] = f"{d[0][-8:].mean():+.2f}"; M["SpecHighIns"] = f"{d[1][-8:].mean():+.2f}"; M["SpecHighTxt"] = f"{d[2][-8:].mean():+.2f}"
    M["SpecLowInp"] = f"{d[0][:8].mean():+.2f}"; M["SpecLowIns"] = f"{d[1][:8].mean():+.2f}"; M["SpecLowTxt"] = f"{d[2][:8].mean():+.2f}"

    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3))
    for h in (1, 2, 3):
        ax = axs[h - 1]
        for j, (k, v) in enumerate(A[f"roc_held{h}"].items()):
            ax.plot(v[0], v[1], label=f"{k} ({A[f'auc_held{h}'][k]:.3f})", color=C(j), lw=1)
        ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=.6); ax.set_title(f"Unseen: {HELD[h-1]}"); ax.set_xlabel("False positive rate"); ax.legend(loc="lower right", fontsize=5)
    axs[0].set_ylabel("True positive rate"); save(fig, "fig_roc.png")

    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2)); bl = ["low", "mid", "high"]
    B = {k: np.concatenate([A[f"band_held{h}"][k] for h in (1, 2, 3)]) for k in ("real", "known", "unseen")}
    for j, (k, col) in enumerate([("real", 0), ("known", 1), ("unseen", 3)]):
        axs[0].bar(np.arange(3) + (j - 1) * .27, B[k].mean(0), .27, yerr=B[k].std(0), capsize=1.5, label=k if k != "known" else "known fakes", color=C(col))
    axs[0].set_xticks(range(3)); axs[0].set_xticklabels(["low band", "mid band", "high band"]); axs[0].set_ylabel("Feature sensitivity $\\|\\Delta f\\|/\\|\\delta\\|$"); axs[0].legend()
    for ax, k, ttl in [(axs[1], "real", "real"), (axs[2], "unseen", "unseen fake")]:
        pass
    allA = {k: np.concatenate([A[f"A_held{h}"][k] for h in (1, 2, 3)]) for k in ("real", "unseen")}
    axs[1].hist(allA["real"], bins=40, alpha=.6, label="real", color=C(0), density=True); axs[1].hist(allA["unseen"], bins=40, alpha=.6, label="unseen fake", color=C(3), density=True)
    axs[1].set_xlabel("A(x) (mean over bands)"); axs[1].set_ylabel("Density"); axs[1].legend()
    allS = {k: np.concatenate([A[f"score_held{h}"][k] for h in (1, 2, 3)]) for k in ("real", "known", "unseen")}
    for k, col, lab in [("real", 0, "real"), ("known", 1, "known fakes"), ("unseen", 3, "unseen fake")]:
        axs[2].hist(allS[k], bins=30, range=(0, 1), alpha=.55, label=lab, color=C(col), density=True)
    axs[2].set_xlabel("Detector fake-probability"); axs[2].set_yscale("log"); axs[2].legend()
    save(fig, "fig_freq.png")
    M["ARealMean"] = f"{allA['real'].mean():.3f}"; M["AUnseenMean"] = f"{allA['unseen'].mean():.3f}"
    M["BandRealLow"], M["BandRealMid"], M["BandRealHigh"] = [f"{v:.3f}" for v in B["real"].mean(0)]
    M["BandUnseenLow"], M["BandUnseenMid"], M["BandUnseenHigh"] = [f"{v:.3f}" for v in B["unseen"].mean(0)]
    # post-hoc diagnostic (exploratory, not part of the pre-set method): AUROC of single-band sensitivities, real vs unseen
    from sklearn.metrics import roc_auc_score as _auc
    bt = "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\nScore (real vs unseen) & Inpaint. & InsightF. & Text2Img & Avg \\\\\n\\midrule\n"
    for nm, fn in [("Low-band sensitivity", lambda b: b[:, 0]), ("Mid-band sensitivity (inverted)", lambda b: -b[:, 1]), ("High-band sensitivity", lambda b: b[:, 2]), ("High / mid ratio", lambda b: b[:, 2] / b[:, 1]), ("Mean over bands, $A(x)$", lambda b: b.mean(1))]:
        vals = []
        for h in (1, 2, 3):
            r_, u_ = A[f"band_held{h}"]["real"], A[f"band_held{h}"]["unseen"]
            vals.append(_auc([0] * len(r_) + [1] * len(u_), np.concatenate([fn(r_), fn(u_)])))
        bt += f"{nm} & " + " & ".join(f"{100*v:.1f}" for v in vals) + f" & {100*np.mean(vals):.1f} \\\\\n"
        if nm.startswith("High /"): M["BandRatioAUC"] = f"{100*np.mean(vals):.1f}"
        if nm.startswith("Mid"): M["BandMidAUC"] = f"{100*np.mean(vals):.1f}"
        if nm.startswith("High-"): M["BandHighAUC"] = f"{100*np.mean(vals):.1f}"
    bt += "\\bottomrule\n\\end{tabular}\n"; open(os.path.join(a.out, "tab_bands.tex"), "w").write(bt)
    M["ScoreUnseenMedian"] = f"{np.median(allS['unseen']):.2f}"; M["ScoreRealMedian"] = f"{np.median(allS['real']):.3f}"
    M["ScoreUnseenBelowHalf"] = f"{100*(allS['unseen'] < .5).mean():.1f}"

    Z, lab = A["pca"]; fig, ax = plt.subplots(figsize=(3.4, 2.6)); names = {0: "real", 1: "inpainting", 2: "insight", 3: "text2img"}
    for k in (0, 2, 3, 1):
        ax.scatter(Z[lab == k, 0], Z[lab == k, 1], s=4, alpha=.5, color=C(k), label=names[k] + (" (unseen)" if k == 1 else ""))
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.legend(markerscale=2); save(fig, "fig_pca.png")

# ---------------------------------------------------------------- sensitivity sweeps
S = {}
for f in sorted(glob.glob(os.path.join(a.sweep, "s*_held*_*__*.json"))):
    r = json.load(open(f)); S.setdefault((r["method"], r["tag"]), []).append(r)
if S:
    lam = {"lam100": 100, "lam10000": 10000, "lam100000": 100000}
    base = [r for r in R["kd_ewc_gem"].values() if r["seed"] == 0]
    s = "\\begin{tabular}{@{}lccc@{}}\n\\toprule\n$\\lambda_{EWC}$ & Closed AUC & Unseen AUROC & BWT \\\\\n\\midrule\n"
    pts = [(100, [r for (m, t), v in S.items() if t == "lam100" for r in v])] if ("kd_ewc_gem", "lam100") in S else []
    L = []
    for t, v in lam.items():
        if ("kd_ewc_gem", t) in S: L.append((v, S[("kd_ewc_gem", t)]))
    L.append((1000, base)); L.sort(key=lambda z: z[0])
    for v, rs in L:
        s += f"{v:,} & {100*np.mean([closed(r) for r in rs]):.1f} & {100*np.mean([r['open_set']['ckm_only'] for r in rs]):.1f} & {100*np.mean([r['bwt'] for r in rs]):+.2f} \\\\\n".replace(",", "{,}")
    s += "\\bottomrule\n\\end{tabular}\n"; open(os.path.join(a.out, "tab_lambda.tex"), "w").write(s)
    s = "\\begin{tabular}{@{}llccc@{}}\n\\toprule\nDirichlet $\\alpha$ & Method & Closed AUC & Unseen AUROC & BWT \\\\\n\\midrule\n"
    al = [("0.1", "alpha0.1"), ("0.5 (default)", None), ("5.0", "alpha5.0")]
    for lab, t in al:
        for nm, m in [("FedAvg", "fedavg"), ("FedOPAW", "full")]:
            rs = base_ = ([r for r in R[m].values() if r["seed"] == 0] if t is None else S.get((m, t), []))
            if not rs: continue
            key = "ckm_only" if m == "fedavg" else "fusion_full_prov"
            s += f"{lab if nm=='FedAvg' else ''} & {nm} & {100*np.mean([closed(r) for r in rs]):.1f} & {100*np.mean([r['open_set'][key] for r in rs]):.1f} & {100*np.mean([r['bwt'] for r in rs]):+.2f} \\\\\n"
    s += "\\bottomrule\n\\end{tabular}\n"; open(os.path.join(a.out, "tab_alpha.tex"), "w").write(s)
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.2))
    axs[0].errorbar([np.log10(v) for v, _ in L], [100 * np.mean([closed(r) for r in rs]) for _, rs in L], fmt="o-", ms=3, color=C(0), label="closed AUC")
    ax2 = axs[0].twinx(); ax2.plot([np.log10(v) for v, _ in L], [100 * np.mean([r["bwt"] for r in rs]) for _, rs in L], "s--", ms=3, color=C(3), label="BWT"); ax2.set_ylabel("BWT (points)", color=C(3))
    axs[0].set_xlabel("$\\log_{10}\\lambda_{EWC}$"); axs[0].set_ylabel("Closed-set AUC (%)", color=C(0))
    for nm, m, c in [("FedAvg", "fedavg", 1), ("FedOPAW", "full", 2)]:
        xs, ys = [], []
        for lab, t in [(0.1, "alpha0.1"), (0.5, None), (5.0, "alpha5.0")]:
            rs = [r for r in R[m].values() if r["seed"] == 0] if t is None else S.get((m, t), [])
            if rs: xs.append(lab); ys.append(100 * np.mean([closed(r) for r in rs]))
        axs[1].plot(xs, ys, "o-", ms=3, color=C(c), label=nm)
    axs[1].set_xscale("log"); axs[1].set_xlabel("Dirichlet $\\alpha$ (smaller = more skewed client sizes)"); axs[1].set_ylabel("Closed-set AUC (%)"); axs[1].legend()
    save(fig, "fig_sweeps.png")

with open(os.path.join(a.out, "extras.tex"), "w") as fh:
    for k, v in M.items(): fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
for name, mean, p in sig: print(f"{name:55s} {mean:+.2f}  p={p:.3f}")
print("macros", len(M), "figures done")
