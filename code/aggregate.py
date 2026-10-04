"""Turn results/*.json into LaTeX tables, figures and a numbers file. Nothing here is typed by hand:
the manuscript \\input's the generated tables so a quoted number always equals the measured one.
Usage: python aggregate.py --results results --out generated
"""
import argparse, glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results"); ap.add_argument("--out", default="generated"); ap.add_argument("--n_ident", type=int, default=3000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

R = {}  # method -> list of result dicts (one per seed x held-out family)
for f in sorted(glob.glob(os.path.join(a.results, "s*_held*_*.json"))):
    if "__" in os.path.basename(f): continue  # sensitivity runs are aggregated in extras.py
    r = json.load(open(f)); R.setdefault(r["method"], []).append(r)
WM = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.results, "wm_s*_held*.json")))]
nfold = {m: len(v) for m, v in R.items()}
HELD = ["inpainting", "insight", "text2img"]

def ms(vals, pct=True, d=1):
    v = np.array([x for x in vals if x is not None and not np.isnan(x)], float)
    if len(v) == 0: return "--"
    k = 100 if pct else 1
    return f"{v.mean()*k:.{d}f}" + (f"$\\pm${v.std(ddof=1)*k:.{d}f}" if len(v) > 1 else "")

def col(m, fn): return [fn(r) for r in R.get(m, [])]
def closed(r): return (r["auc_t1_end"] + r["auc_t2_end"]) / 2
def rounds_to(r, thr):
    for i, v in enumerate(r["val_curve"]):
        if v >= thr: return i + 1
    return None

nums = {"n_runs": nfold}
central_auc = np.mean(col("central", closed)) if "central" in R else float("nan")
nums["central_closed_auc"] = central_auc
T = {}

# ---- Table: closed-set detection
rows = [("Centralized SpecFuse", "central"), ("FedAvg", "fedavg"), ("FedProx", "fedprox"),
        ("FedAvg+EWC (local Fisher)", "fedavg_ewc"), ("\\textbf{FedOPAW}", "full")]
s = "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\nMethod & Task 1 & Task 2 & Avg & Gap \\\\\n\\midrule\n"
for name, m in rows:
    if m not in R: continue
    gap = [closed(r) - central_auc for r in R[m]]
    s += f"{name} & {ms(col(m, lambda r: r['auc_t1_end']))} & {ms(col(m, lambda r: r['auc_t2_end']))} & {ms(col(m, closed))} & {ms(gap, d=2) if m!='central' else '--'} \\\\\n"
s += "\\bottomrule\n\\end{tabular}\n"
T["closed"] = s

# ---- Table: open-set AUROC per held-out generator
def os_by_held(m, key):
    out = {}
    for r in R.get(m, []):
        v = r["open_set"].get(key)
        if v is not None: out.setdefault(r["held"], []).append(v)
    return out
orows = [("CLIP zero-shot", "fedavg", "clip_zeroshot"), ("UnivFD-style probe", "fedavg", "univfd_style"),
         ("FreqAnom alone", "kd_ewc_gem", "freqanom_only"), ("FedOPAW (detector only)", "kd_ewc_gem", "ckm_only"),
         ("FedOPAW (+CLIP)", "kd_ewc_gem", "fusion_clip"), ("FedOPAW (+FreqAnom)", "kd_ewc_gem", "fusion_freq"),
         ("FedOPAW (full fusion)", "kd_ewc_gem", "fusion_full"), ("\\textbf{FedOPAW + provenance}", "full", "fusion_full_prov")]
s = "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\nMethod & Inpaint. & InsightF. & Text2Img & Avg \\\\\n\\midrule\n"
os_avg = {}
for name, m, key in orows:
    d = os_by_held(m, key)
    if not d: continue
    per = [np.mean(d[h]) if h in d else None for h in HELD]
    avg = [x for x in per if x is not None]
    os_avg[key + "@" + m] = float(np.mean(avg)) if avg else None
    s += f"{name} & " + " & ".join(ms([x]) for x in per) + f" & {ms(avg)} \\\\\n"
s += "\\bottomrule\n\\end{tabular}\n"
T["openset"] = s; nums["openset_avg"] = os_avg

# ---- Table: continual learning
s = "\\begin{tabular}{@{}lccc@{}}\n\\toprule\nMethod & BWT (pts) & Task-1 retention (\\%) & Rounds to 95\\% cen. \\\\\n\\midrule\n"
for name, m in [("FedAvg", "fedavg"), ("FedProx", "fedprox"), ("FedAvg+EWC", "fedavg_ewc"), ("FedAvg+KD", "kd"),
                ("FedAvg+KD+Fed-EWC", "kd_ewc"), ("\\textbf{FedOPAW}", "full")]:
    if m not in R: continue
    ret = col(m, lambda r: r["auc_t1_end"] / r["auc_t1_mid"])
    rr = col(m, lambda r: rounds_to(r, 0.95 * central_auc))
    okr = [x for x in rr if x is not None]
    s += f"{name} & {ms(col(m, lambda r: r['bwt']), d=2).replace('$\\pm$','$\\pm$')} & {ms(ret)} & " + \
         (f"{np.mean(okr):.1f} ({len(okr)}/{len(rr)})" if okr else "never") + " \\\\\n"
s = s.replace("BWT (pts)", "BWT (AUC)")
s += "\\bottomrule\n\\end{tabular}\n"
T["continual"] = s

# ---- Table: communication
s = "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\nMethod & Uplink & Compr. & Rounds to & Final \\\\\n& (MB/client/rd) & ratio & 90\\% AUC & AUC \\\\\n\\midrule\n"
base_up = np.mean([np.mean(r["uplink_mb"]) for r in R.get("fedavg", [])]) if "fedavg" in R else float("nan")
for name, m in [("FedAvg (fp32 weights)", "fedavg"), ("FedOPAW (int8 update)", "full"), ("FedOPAW + 30\\% sparse mask", "full_sparse")]:
    if m not in R: continue
    up = np.mean([np.mean(r["uplink_mb"]) for r in R[m]])
    rr = [rounds_to(r, 0.90) for r in R[m]]; okr = [x for x in rr if x is not None]
    s += f"{name} & {up:.1f} & {base_up/up:.1f}$\\times$ & " + (f"{np.mean(okr):.1f}" if okr else "never") + f" & {ms(col(m, closed))} \\\\\n"
s += "\\bottomrule\n\\end{tabular}\n"
T["comm"] = s

# ---- Table: ablation (cumulative)
s = "\\begin{tabular}{@{}lccc@{}}\n\\toprule\nConfiguration & Closed AUC & Open-set AUROC & BWT \\\\\n\\midrule\n"
abl = [("FedAvg only", "fedavg", "ckm_only"), ("+ Federated KD", "kd", "ckm_only"), ("+ Fed-EWC", "kd_ewc", "ckm_only"),
       ("+ GEM projection", "kd_ewc_gem", "ckm_only"), ("+ CLIP zero-shot head", "kd_ewc_gem", "fusion_clip"),
       ("+ FreqAnom scorer", "kd_ewc_gem", "fusion_freq"), ("+ Calibrated fusion", "kd_ewc_gem", "fusion_full"),
       ("+ Provenance $\\tau$ (+int8 update)", "full", "fusion_full_prov")]
for name, m, key in abl:
    if m not in R: continue
    o = [np.mean(v) for v in os_by_held(m, key).values()]
    s += f"{name} & {ms(col(m, closed))} & {ms(o)} & {ms(col(m, lambda r: r['bwt']), d=2)} \\\\\n"
s += "\\bottomrule\n\\end{tabular}\n"
T["ablation"] = s

# ---- Table: provenance FPR per tau
s = "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\nAuthentic-image FPR (\\%) & $\\tau$=0 & $\\tau$=0.3 & $\\tau$=1.0 & Overall \\\\\n\\midrule\n"
for name, m in [("FedOPAW without $\\tau$", "kd_ewc_gem"), ("FedOPAW with $\\tau$", "full")]:
    if m not in R: continue
    cells = []
    for l in ("0.0", "0.3", "1.0"):
        cells.append(ms([r["tau_fpr"][l] for r in R[m]]))
    # overall FPR from the two count-weighted sums
    ov = []
    for r in R[m]:
        n = r["tau_fpr"]["n"]; tot = sum(n.values())
        ov.append(sum((r["tau_fpr"][l] or 0) * n[l] for l in n) / tot)
    s += f"{name} & " + " & ".join(cells) + f" & {ms(ov)} \\\\\n"
s += "\\bottomrule\n\\end{tabular}\n"
T["prov"] = s

# ---- Table: watermark
if WM:
    s = "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\nDegradation & \\multicolumn{2}{c}{WBA (\\%)} & \\multicolumn{4}{c}{$\\Delta$AUC (pts)} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-7}\n & naive & AdvMark & naive & det.-only & emb.-only & AdvMark \\\\\n\\midrule\n"
    lab = {"clean": "Clean", "gauss": "Gauss. $\\sigma$=0.05", "jpeg": "JPEG Q50", "crop": "20\\% crop"}
    for k in ("clean", "gauss", "jpeg", "crop"):
        g = lambda who, f: [w["rows"][k][who][f] for w in WM]
        s += f"{lab[k]} & {ms(g('naive','wba_authentic'), False)} & {ms(g('advmark','wba_authentic'), False)} & {ms(g('naive','delta_auc'), False)} & {ms(g('det_ft','delta_auc'), False)} & {ms(g('emb_only','delta_auc'), False)} & {ms(g('advmark','delta_auc'), False)} \\\\\n"
    s += "\\bottomrule\n\\end{tabular}\n"
    T["wm"] = s
    nums["conflict_mean"] = {w_: float(np.mean([x["conflict"][w_]["mean"] for x in WM])) for w_ in ("naive", "advmark")}
    nums["conflict_frac_below_-0.5"] = {w_: float(np.mean([x["conflict"][w_]["frac_below_minus_half"] for x in WM])) for w_ in ("naive", "advmark")}
    nums["psnr"] = {"naive": float(np.mean([x["psnr_naive"] for x in WM])), "advmark": float(np.mean([x["psnr_advmark"] for x in WM]))}

for k, v in T.items():
    open(os.path.join(a.out, f"tab_{k}.tex"), "w").write(v)

# ---- figures
plt.rcParams.update({"font.size": 8})
fig, ax = plt.subplots(figsize=(3.4, 2.4))
for m, lab, c in [("fedavg", "FedAvg", "tab:orange"), ("fedprox", "FedProx", "tab:purple"), ("kd", "FedAvg+KD", "tab:green"), ("full", "FedOPAW", "tab:blue")]:
    if m in R:
        L = min(len(r["val_curve"]) for r in R[m]); Y = np.mean([r["val_curve"][:L] for r in R[m]], 0)
        ax.plot(range(1, L + 1), Y * 100, label=lab, color=c)
ax.axhline(0.95 * central_auc * 100, ls=":", color="gray", label="95% of centralized")
R_ = max(len(r["val_curve"]) for m in R for r in R[m])
ax.axvline(R_ // 2 + 0.5, ls="--", color="k", lw=0.6); ax.text(R_ // 2 + 0.7, ax.get_ylim()[0] + 0.5, "task 2 starts", fontsize=6)
ax.set_xlabel("Global round"); ax.set_ylabel("Validation AUC (%)"); ax.legend(fontsize=6); fig.tight_layout()
fig.savefig(os.path.join(a.out, "fig_convergence.png"), dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(3.4, 2.4)); w = 0.16
labs = ["CLIP zs", "UnivFD-style", "FreqAnom", "CKM only", "Fusion", "Fusion+$\\tau$"]
keys = [("fedavg", "clip_zeroshot"), ("fedavg", "univfd_style"), ("kd_ewc_gem", "freqanom_only"), ("kd_ewc_gem", "ckm_only"), ("kd_ewc_gem", "fusion_full"), ("full", "fusion_full_prov")]
for j, (lb, (m, k)) in enumerate(zip(labs, keys)):
    d = os_by_held(m, k)
    ax.bar(np.arange(3) + (j - 2.5) * w, [np.mean(d[h]) * 100 if h in d else 0 for h in HELD], w, label=lb)
ax.set_xticks(range(3)); ax.set_xticklabels(["hold out\ninpainting", "hold out\ninsight", "hold out\ntext2img"])
ax.set_ylabel("Unseen-family AUROC (%)"); ax.set_ylim(40, 100); ax.legend(fontsize=5, ncol=2); fig.tight_layout()
fig.savefig(os.path.join(a.out, "fig_openset.png"), dpi=200); plt.close(fig)

if WM:
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for who, c in (("naive", "tab:red"), ("advmark", "tab:blue")):
        ax.hist(np.concatenate([x["conflict"][who]["values"] for x in WM]), bins=40, alpha=0.55, color=c, label=who)
    ax.axvline(-0.5, ls=":", color="k"); ax.set_xlabel("cos($\\nabla L_{wm}$, $\\nabla L_{det}$)"); ax.set_ylabel("Samples"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_conflict.png"), dpi=200); plt.close(fig)

json.dump(nums, open(os.path.join(a.out, "numbers.json"), "w"), indent=1)
print(json.dumps(nums, indent=1)); print("tables:", list(T))

# ---- LaTeX macros so prose numbers are generated, not typed
def mean_of(vals): 
    v=[x for x in vals if x is not None and not np.isnan(x)]; return float(np.mean(v)) if v else float("nan")
M = {}
for k, m in [("Central","central"),("FedAvg","fedavg"),("FedProx","fedprox"),("FedAvgEWC","fedavg_ewc"),("KD","kd"),("KDEWC","kd_ewc"),("KDEWCGEM","kd_ewc_gem"),("Full","full"),("FullSparse","full_sparse")]:
    if m in R:
        M["Closed"+k] = f"{100*mean_of(col(m, closed)):.1f}"
        M["BWT"+k] = f"{mean_of(col(m, lambda r: r['bwt'])):+.3f}"
        if m != "central": M["Up"+k] = f"{mean_of([np.mean(r['uplink_mb']) for r in R[m]]):.1f}"
for k, key in [("Clip","clip_zeroshot@fedavg"),("Univfd","univfd_style@fedavg"),("Freq","freqanom_only@kd_ewc_gem"),("Ckm","ckm_only@kd_ewc_gem"),("FusClip","fusion_clip@kd_ewc_gem"),("FusFreq","fusion_freq@kd_ewc_gem"),("Fusion","fusion_full@kd_ewc_gem"),("FusionProv","fusion_full_prov@full")]:
    if os_avg.get(key) is not None: M["OS"+k] = f"{100*os_avg[key]:.1f}"
if "fedavg" in R and "full" in R:
    M["RatioInt"] = f"{mean_of([np.mean(r['uplink_mb']) for r in R['fedavg']])/mean_of([np.mean(r['uplink_mb']) for r in R['full']]):.1f}"
if "full_sparse" in R:
    M["RatioSparse"] = f"{mean_of([np.mean(r['uplink_mb']) for r in R['fedavg']])/mean_of([np.mean(r['uplink_mb']) for r in R['full_sparse']]):.1f}"
for m_, nm in (("kd_ewc_gem","NoTau"),("full","Tau")):
    if m_ in R:
        M["FPR"+nm] = f"{100*mean_of([sum((r['tau_fpr'][l] or 0)*r['tau_fpr']['n'][l] for l in r['tau_fpr']['n'])/sum(r['tau_fpr']['n'].values()) for r in R[m_]]):.1f}"
        M["FPRone"+nm] = f"{100*mean_of([r['tau_fpr']['1.0'] for r in R[m_]]):.1f}"
        M["FPRzero"+nm] = f"{100*mean_of([r['tau_fpr']['0.0'] for r in R[m_]]):.1f}"
if WM:
    for who in ("naive","det_ft","emb_only","advmark"):
        for kind in ("clean","gauss","jpeg","crop"):
            M[f"WM{who.replace('_','')}{kind}"] = f"{mean_of([w['rows'][kind][who]['delta_auc'] for w in WM]):+.1f}"
    for kind in ("clean","jpeg","crop"):
        M[f"WBAnaive{kind}"] = f"{mean_of([w['rows'][kind]['naive']['wba_authentic'] for w in WM]):.1f}"
        M[f"WBAadv{kind}"] = f"{mean_of([w['rows'][kind]['advmark']['wba_authentic'] for w in WM]):.1f}"
    M["ConflictNaive"] = f"{nums['conflict_mean']['naive']:+.4f}"; M["ConflictAdv"] = f"{nums['conflict_mean']['advmark']:+.4f}"
    M["PSNR"] = f"{nums['psnr']['naive']:.1f}"
gaps = [100*(central_auc - mean_of(col(m, closed))) for m in ("fedavg","fedprox","fedavg_ewc","full") if m in R]
M["GapFedMin"] = f"{min(gaps):.1f}"; M["GapFedMax"] = f"{max(gaps):.1f}"
if "full" in R and "full_sparse" in R: M["SparseDrop"] = f"{100*(mean_of(col('full', closed)) - mean_of(col('full_sparse', closed))):.1f}"
M["NIdent"] = f"{a.n_ident:,}".replace(",", "{,}"); M["NImgs"] = f"{4*a.n_ident:,}".replace(",", "{,}")
M["NRounds"] = str(max(r["hp"]["rounds"] for m in R for r in R[m]))
M["NSeeds"] = str(len({r["seed"] for m in R for r in R[m]})); M["NFolds"] = str(len({r["held"] for m in R for r in R[m]}))
M["NRuns"] = str(max(len(v) for v in R.values()))
with open(os.path.join(a.out, "numbers.tex"), "w") as fh:
    for k, v in M.items(): fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
print("macros:", len(M))
