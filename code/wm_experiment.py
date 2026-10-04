"""Watermark / detector co-design pilot (AdvMark-Fed proxy) on top of a trained FedOPAW global model.

Disclosed simplifications (all real, none faked):
  * embedder is a learnable spread-spectrum embedder (48 mid-band patterns) with a correlation decoder, not the diffusion-based DiffMark; a first HiDDeN-style CNN failed to learn the message at this scale and was discarded
  * "manipulated watermarked content" x_w^fake is a proxy: the generator's fake image plus the
    watermark residual r(x_real) of the authentic source (the mark carried over the manipulation).
    We do not re-run the generators on watermarked inputs.
  * the detector is frozen during embedder fine-tuning (the paper also updates its last two layers)
  * degradations: Gaussian noise 0.05, JPEG q50, 20% crop-resize. H.264 is not evaluated (image data).
Usage: python wm_experiment.py --data data/dff_n3000_s128.npz --model results/s0_held1_full.pt --held 1
"""
import argparse, io, json, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
import fedopaw_pilot as P

MSG = 48

def band_noise(n, S, lo=0.04, hi=0.35, seed=0):
    """Unit-variance patterns with energy in a mid-frequency band (survives JPEG/noise better than white)."""
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(S), torch.arange(S), indexing="ij")
    r = torch.sqrt((yy - S // 2) ** 2 + (xx - S // 2) ** 2) / (S / 2)
    mask = ((r >= lo) & (r < hi)).float()
    z = torch.randn(n, 3, S, S, generator=g)
    Fz = torch.fft.fftshift(torch.fft.fft2(z), dim=(-2, -1)) * mask
    p = torch.fft.ifft2(torch.fft.ifftshift(Fz, dim=(-2, -1))).real
    return p / p.flatten(1).std(dim=1).view(-1, 1, 1, 1)

class Enc(nn.Module):
    """Learnable spread-spectrum embedder: residual = alpha/sqrt(MSG) * sum_i (2m_i-1) P_i (48-bit message)."""
    def __init__(s, S=128, alpha=0.03):
        super().__init__(); s.alpha = alpha
        s.P = nn.Parameter(band_noise(MSG, S))
    def residual(s, x, m):
        return s.alpha / MSG ** 0.5 * torch.einsum("bk,kchw->bchw", 2 * m - 1, s.P)
    def forward(s, x, m):
        return (x + s.residual(x, m)).clamp(0, 1)

class Dec(nn.Module):
    """Correlation decoder against the embedder's patterns (its own learnable copy) plus a learnable gain."""
    def __init__(s, S=128):
        super().__init__()
        s.P = nn.Parameter(band_noise(MSG, S))
        s.gain = nn.Parameter(torch.tensor(300.0))
    def forward(s, x):
        xc = x - x.mean((1, 2, 3), keepdim=True)
        return s.gain * torch.einsum("bchw,kchw->bk", xc, s.P) / s.P[0].numel()

def degrade(x, kind, g):
    if kind == "clean":
        return x
    if kind == "gauss":
        return (x + 0.05 * torch.randn(x.shape, generator=g).to(x.device)).clamp(0, 1)
    if kind == "crop":  # 20% crop then resize back
        S = x.shape[-1]; c = int(0.1 * S)
        return F.interpolate(x[:, :, c:S - c, c:S - c], size=S, mode="bilinear", align_corners=False)
    if kind == "jpeg":
        out = []
        for im in (x.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8):
            b = io.BytesIO(); Image.fromarray(im).save(b, "JPEG", quality=50); b.seek(0)
            out.append(np.asarray(Image.open(b).convert("RGB")))
        return torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).float().div(255).to(x.device)
    raise ValueError(kind)

def train_embedder(enc, dec, D, idx, steps, det=None, lam_adv=0.0, known=None, lr=1e-3, log=print, det_opt=None, train_emb=True):
    ps = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(ps, lr=lr); g = torch.Generator().manual_seed(1)
    for t in range(steps):
        sel = np.random.choice(idx, 24, replace=False)
        x = torch.stack([D.x[i, 0] for i in sel]).float().div(255).to(P.DEV)
        m = (torch.rand(len(x), MSG) > 0.5).float().to(P.DEV)
        xw = enc(x, m)
        kind = ["clean", "gauss", "crop"][t % 3]
        l_wm = F.binary_cross_entropy_with_logits(dec(degrade(xw, kind, g)), m)
        l_img = F.mse_loss(xw, x)
        loss = l_wm + 5 * l_img  # image term is constant here (fixed alpha); kept for logging
        if det is not None and lam_adv > 0:
            fam = known[np.random.randint(2)]
            fk = torch.stack([D.x[i, fam] for i in sel]).float().div(255).to(P.DEV)
            xf = (fk + enc.residual(x, m)).clamp(0, 1)  # carried-over mark on the manipulated image
            z = torch.zeros(len(x), device=P.DEV)
            l_det = F.binary_cross_entropy_with_logits(det(xw, z), torch.zeros(len(x), device=P.DEV)) + \
                    F.binary_cross_entropy_with_logits(det(xf, z), torch.ones(len(x), device=P.DEV))
            loss = loss + lam_adv * l_det
        if train_emb:
            opt.zero_grad(); loss.backward(); opt.step()
        if det_opt is not None:  # alternate step: update the detector's last two layers on watermarked data
            with torch.no_grad():
                xw_d = enc(x, m); xf_d = (fk + enc.residual(x, m)).clamp(0, 1)
            z = torch.zeros(len(x), device=P.DEV)
            ld = F.binary_cross_entropy_with_logits(det(xw_d, z), torch.zeros(len(x), device=P.DEV)) + \
                 F.binary_cross_entropy_with_logits(det(xf_d, z), torch.ones(len(x), device=P.DEV))
            det_opt.zero_grad(); ld.backward(); det_opt.step()
        if t % 50 == 0:
            log(f"   step {t} wm={l_wm.item():.3f} img={l_img.item():.5f}" + (f" det={l_det.item():.3f}" if det is not None and lam_adv > 0 else ""))

@torch.no_grad()
def detect(det, xs, bs=128):
    det.eval(); out = []
    for k in range(0, len(xs), bs):
        out.append(torch.sigmoid(det(xs[k:k + bs].to(P.DEV), torch.zeros(len(xs[k:k + bs]), device=P.DEV))).cpu())
    return torch.cat(out).numpy()

def conflict(det, enc, dec, xs, ys, ms):
    """cos( grad_x L_wm , grad_x L_det ) per sample, evaluated on watermarked inputs (Eq. 15)."""
    cs = []
    det.eval(); dec.eval(); enc.eval()
    for k in range(len(xs)):
        x = xs[k:k + 1].to(P.DEV).clone().requires_grad_(True)
        l_wm = F.binary_cross_entropy_with_logits(dec(x), ms[k:k + 1].to(P.DEV))
        gw = torch.autograd.grad(l_wm, x)[0].flatten()
        x2 = xs[k:k + 1].to(P.DEV).clone().requires_grad_(True)
        l_d = F.binary_cross_entropy_with_logits(det(x2, torch.zeros(1, device=P.DEV)), ys[k:k + 1].to(P.DEV))
        gd = torch.autograd.grad(l_d, x2)[0].flatten()
        cs.append(float(F.cosine_similarity(gw, gd, dim=0)))
    return np.array(cs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--held", type=int, required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300); ap.add_argument("--adv_steps", type=int, default=400)
    ap.add_argument("--lam_adv", type=float, default=3.0); ap.add_argument("--out", default="results")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    D = P.Data(a.data, a.seed); known = [s for s in (1, 2, 3) if s != a.held]
    det = P.new_model(D); det.load_state_dict(torch.load(a.model, map_location=P.DEV)); det.eval()
    for p in det.parameters(): p.requires_grad_(False)
    print("training naive embedder"); enc, dec = Enc(D.S).to(P.DEV), Dec(D.S).to(P.DEV)
    train_embedder(enc, dec, D, D.train, a.steps)
    naive = (Enc(D.S).to(P.DEV), Dec(D.S).to(P.DEV))
    naive[0].load_state_dict(enc.state_dict()); naive[1].load_state_dict(dec.state_dict())
    print("adversarial fine-tune (AdvMark-Fed proxy)")
    adv = (Enc(D.S).to(P.DEV), Dec(D.S).to(P.DEV))
    adv[0].load_state_dict(enc.state_dict()); adv[1].load_state_dict(dec.state_dict())
    import copy
    def last_two(d):
        for p in d.parameters(): p.requires_grad_(False)
        ps = list(d.fuse.parameters()) + list(d.head.parameters())
        for p in ps: p.requires_grad_(True)
        return torch.optim.Adam(ps, lr=1e-4)
    # (a) AdvMark-Fed: embedder and detector last two layers updated alternately
    det_adv = copy.deepcopy(det); opt_a = last_two(det_adv)
    train_embedder(adv[0], adv[1], D, D.train, a.adv_steps, det=det_adv, lam_adv=a.lam_adv, known=known, lr=1e-2, det_opt=opt_a)
    # (b) detector-only adaptation: naive embedder frozen, detector last two layers updated
    det_ft = copy.deepcopy(det); opt_b = last_two(det_ft)
    train_embedder(copy.deepcopy(naive[0]), copy.deepcopy(naive[1]), D, D.train, a.adv_steps, det=det_ft, lam_adv=a.lam_adv, known=known, lr=1e-2, det_opt=opt_b, train_emb=False)
    for d_ in (det_adv, det_ft):
        for p in d_.parameters(): p.requires_grad_(False)
        d_.eval()
    # (c) embedder-only fine-tune against the frozen detector (kept as an ablation)
    adv_frozen = (Enc(D.S).to(P.DEV), Dec(D.S).to(P.DEV))
    adv_frozen[0].load_state_dict(enc.state_dict()); adv_frozen[1].load_state_dict(dec.state_dict())
    train_embedder(adv_frozen[0], adv_frozen[1], D, D.train, a.adv_steps, det=det, lam_adv=a.lam_adv, known=known, lr=1e-2)
    VAR = (("naive", naive, det), ("det_ft", naive, det_ft), ("advmark", adv, det_adv), ("emb_only", adv_frozen, det))

    test = D.test[:300]
    real = torch.stack([D.x[i, 0] for i in test]).float().div(255)
    fakes = [torch.stack([D.x[i, f] for i in test]).float().div(255) for f in known]
    fake = torch.cat(fakes); nreal = len(real)
    y = [0] * nreal + [1] * len(fake)
    g = torch.Generator().manual_seed(3)
    msg = (torch.rand(nreal, MSG) > 0.5).float()
    res = {"held": P.SRC[a.held], "known": [P.SRC[k] for k in known], "rows": {}}
    for kind in ["clean", "gauss", "jpeg", "crop"]:
        row = {}
        base = detect(det, torch.cat([degrade(real.to(P.DEV), kind, g), degrade(fake.to(P.DEV), kind, g)]))
        row["auc_no_wm"] = P.auc(y, base)
        for name, (e, d), dt in VAR:
            e.eval(); d.eval()
            with torch.no_grad():
                m = msg.to(P.DEV); xr = real.to(P.DEV)
                xw = e(xr, m); r = xw - xr
                xf = (fake.to(P.DEV) + torch.cat([r] * len(known))).clamp(0, 1)
                dr, df = degrade(xw, kind, g), degrade(xf, kind, g)
                wba = float(((d(dr) > 0).float() == m).float().mean()) * 100
                wba_f = float(((d(df[:nreal]) > 0).float() == m).float().mean()) * 100
            a_ = P.auc(y, detect(dt, torch.cat([dr, df])))
            row[name] = dict(auc=a_, delta_auc=(a_ - row["auc_no_wm"]) * 100, wba_authentic=wba, wba_manipulated=wba_f)
        res["rows"][kind] = row
        print(kind, json.dumps(row))
    # gradient conflict on clean watermarked inputs
    cf = {}
    for name, (e, d), dt in VAR:
        if name not in ("naive", "advmark"): continue
        e.eval()
        with torch.no_grad():
            m = msg.to(P.DEV); xw = e(real.to(P.DEV), m); xf = (fake[:nreal].to(P.DEV) + (xw - real.to(P.DEV))).clamp(0, 1)
        xs = torch.cat([xw[:150], xf[:150]]).cpu(); ys = torch.tensor([0.] * 150 + [1.] * 150)
        ms = torch.cat([msg[:150], msg[:150]])
        c = conflict(dt, e, d, xs, ys, ms)
        cf[name] = dict(mean=float(c.mean()), frac_below_minus_half=float((c < -0.5).mean()), values=c.tolist())
    res["conflict"] = cf
    with torch.no_grad():
        xw = naive[0](real[:100].to(P.DEV), msg[:100].to(P.DEV))
        res["psnr_naive"] = float(-10 * torch.log10(F.mse_loss(xw, real[:100].to(P.DEV))))
        xw = adv[0](real[:100].to(P.DEV), msg[:100].to(P.DEV))
        res["psnr_advmark"] = float(-10 * torch.log10(F.mse_loss(xw, real[:100].to(P.DEV))))
    json.dump(res, open(os.path.join(a.out, f"wm_s{a.seed}_held{a.held}.json"), "w"), indent=1)
    print("saved; conflict means", {k: v["mean"] for k, v in cf.items()}, "psnr", res["psnr_naive"], res["psnr_advmark"])

if __name__ == "__main__":
    main()
