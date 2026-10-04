"""Fetch a paired random subset of DeepFakeFace (OpenRL/DeepFakeFace, public on HuggingFace).
Each identity file exists in wiki (real), inpainting, insight, text2img, so a stem gives one real
image and three fakes of the same person. Range-requests pull only the chosen files from the zips.
Images are centre-cropped (no resampling, to keep generator artifacts) and cached as one npz.
Usage: python fetch_dff.py --n 2000 --size 128 --out data/dff_n2000_s128.npz
"""
import argparse, io, random
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from remotezip import RemoteZip

BASE = "https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/main/"
SRC = ["wiki", "inpainting", "insight", "text2img"]

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--size", type=int, default=128)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="data/dff.npz")
ap.add_argument("--zipdir", default=None, help="folder with the four downloaded zips (skips remote range requests)")
a = ap.parse_args()
import zipfile
def Z(src):
    return zipfile.ZipFile(f"{a.zipdir}/{src}.zip") if a.zipdir else RemoteZip(BASE + src + ".zip")

with Z("wiki") as z:
    stems = sorted(i.filename.split("/", 1)[1] for i in z.infolist() if not i.is_dir())
random.Random(a.seed).shuffle(stems)
stems = stems[: a.n]

def crop(b, s):
    im = Image.open(io.BytesIO(b)).convert("RGB")
    w, h = im.size
    l, t = (w - s) // 2, (h - s) // 2
    return np.asarray(im.crop((l, t, l + s, t + s)), dtype=np.uint8), (w, h)

import time, sys
NCH = 4 if not a.zipdir else 1  # chunks per source -> 4*8 = 32 concurrent connections
def work(job):
    src, c = job
    mine = list(range(c, len(stems), NCH))
    out = {}; z = Z(src)
    for n, k in enumerate(mine):
        st = stems[k]
        for attempt in range(25):
            try:
                out[k] = crop(z.read(f"{src}/{st}"), a.size)[0]; break
            except Exception as e:
                time.sleep(1 + min(attempt, 10) + np.random.rand() * 3)
                try: z = Z(src)
                except Exception: pass
        else:
            raise RuntimeError(f"failed {src}/{st}")
        if n % 100 == 0: print(src, c, n, len(mine), flush=True)
    return src, out

res = {s: {} for s in SRC}
with ThreadPoolExecutor(4 * NCH) as ex:
    for src, out in ex.map(work, [(s, c) for s in SRC for c in range(NCH)]):
        res[src].update(out)
res = {s: np.stack([res[s][k] for k in range(len(stems))]) for s in SRC}
arr = np.stack([res[s] for s in SRC], 1)  # N x 4 x S x S x 3
np.savez_compressed(a.out, x=arr, stems=np.array(stems), sources=np.array(SRC))
print("saved", a.out, arr.shape)
