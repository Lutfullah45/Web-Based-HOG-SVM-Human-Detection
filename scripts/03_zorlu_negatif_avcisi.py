"""
scripts/mine_hard_negatives.py
Train/neg üzerinde hard-negative mining (junction/symlink gerektirmez).

- --proj_root: proje kökü (baseline_work burada)
- --data_root: veri kökü (Train/Test burada)

Örnek:
python .\scripts\mine_hard_negatives.py --prefer_imageio --max_images 200 --score_thr 1.5 --top_k 5 --allow_overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import joblib
import numpy as np

try:
    import imageio.v3 as iio
    _HAS_IMAGEIO = True
except Exception:
    _HAS_IMAGEIO = False

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def imread_bgr(path: Path, prefer_imageio: bool):
    if prefer_imageio and _HAS_IMAGEIO:
        img = iio.imread(str(path))
        if img is None:
            return None
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]
        if img.dtype != np.uint8:
            if np.issubdtype(img.dtype, np.floating):
                img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            elif img.dtype == np.uint16:
                img = (img / 256).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        return img[..., ::-1].copy()
    return cv2.imread(str(path))


def imwrite_bgr(path: Path, img_bgr: np.ndarray):
    # Unicode-safe: cv2.imencode + write_bytes
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError("cv2.imencode başarısız.")
    path.write_bytes(buf.tobytes())


def build_hog_from_meta(meta_path: Path, fallback_win=(64, 128)):
    win_w, win_h = fallback_win
    cell, block_cells, nbins = 8, 2, 9
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            hogm = meta.get("hog", {})
            win_w = int(hogm.get("win_w", win_w))
            win_h = int(hogm.get("win_h", win_h))
            cell = int(hogm.get("cell", cell))
            block_cells = int(hogm.get("block_cells", block_cells))
            nbins = int(hogm.get("nbins", nbins))
        except Exception:
            pass
    block = (cell * block_cells, cell * block_cells)
    hog = cv2.HOGDescriptor(
        _winSize=(win_w, win_h),
        _blockSize=block,
        _blockStride=(cell, cell),
        _cellSize=(cell, cell),
        _nbins=nbins,
    )
    return hog, win_w, win_h


def hog_feat(hog, patch_bgr):
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    return hog.compute(gray).reshape(-1)  # type: ignore


def nms(boxes, scores, iou_thr=0.3):
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2-x1)*(y2-y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1)
        h = np.maximum(0.0, yy2-yy1)
        inter = w*h
        iouv = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iouv <= iou_thr)[0]
        order = order[inds + 1]
    return keep


def mine_one_image(img0, clf, hog, win_w, win_h, score_thr, stride, scale_factor, nms_iou_thr, top_k):
    boxes, scores, patches = [], [], []
    scale = 1.0
    img = img0
    while img.shape[1] >= win_w and img.shape[0] >= win_h:
        H, W = img.shape[:2]
        for y in range(0, H - win_h + 1, stride):
            for x in range(0, W - win_w + 1, stride):
                patch = img[y:y + win_h, x:x + win_w]
                s = float(clf.decision_function([hog_feat(hog, patch)])[0])
                if s >= score_thr:
                    xmin = int(x * scale)
                    ymin = int(y * scale)
                    xmax = int((x + win_w) * scale)
                    ymax = int((y + win_h) * scale)
                    boxes.append((xmin, ymin, xmax, ymax))
                    scores.append(s)
                    patches.append(patch.copy())

        scale *= scale_factor
        new_w = int(img0.shape[1] / scale)
        new_h = int(img0.shape[0] / scale)
        if new_w < win_w or new_h < win_h:
            break
        img = cv2.resize(img0, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if not boxes:
        return [], 0, 0

    keep = nms(boxes, scores, iou_thr=nms_iou_thr)
    kept = sorted([(patches[i], scores[i]) for i in keep], key=lambda t: t[1], reverse=True)
    if top_k > 0:
        kept = kept[:top_k]
    return kept, len(boxes), len(keep)


def maybe_clear_dir(d: Path, allow_overwrite: bool):
    d.mkdir(parents=True, exist_ok=True)
    if allow_overwrite:
        for f in d.glob("*"):
            if f.is_file():
                f.unlink()


def _resolve(base: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp)


def main():
    script_dir = Path(__file__).resolve().parent
    default_proj_root = script_dir.parent
    default_data_root = default_proj_root / "data" / "INRIA_raw"

    ap = argparse.ArgumentParser()
    ap.add_argument("--proj_root", type=str, default=str(default_proj_root))
    ap.add_argument("--data_root", type=str, default=str(default_data_root))

    ap.add_argument("--model_path", type=str, default="baseline_work/hog_svm_linear.joblib")
    ap.add_argument("--meta_path", type=str, default="baseline_work/hog_svm_linear_meta.json")

    ap.add_argument("--neg_img_dir", type=str, default="Train/neg")
    ap.add_argument("--out_dir", type=str, default="baseline_work/prepared/neg_hard")
    ap.add_argument("--manifest", type=str, default="baseline_work/prepared/neg_hard_manifest.csv")

    ap.add_argument("--allow_overwrite", action="store_true")
    ap.add_argument("--prefer_imageio", action="store_true")
    ap.add_argument("--max_images", type=int, default=200)
    ap.add_argument("--score_thr", type=float, default=1.5)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--scale_factor", type=float, default=1.25)
    ap.add_argument("--nms_iou_thr", type=float, default=0.3)
    args = ap.parse_args()

    proj_root = Path(args.proj_root)
    data_root = Path(args.data_root)

    model_path = _resolve(proj_root, args.model_path)
    meta_path = _resolve(proj_root, args.meta_path)

    clf = joblib.load(model_path)
    hog, win_w, win_h = build_hog_from_meta(meta_path, fallback_win=(64, 128))

    neg_dir = _resolve(data_root, args.neg_img_dir)
    files = list_images(neg_dir)
    if args.max_images > 0:
        files = files[:args.max_images]

    out_dir = _resolve(proj_root, args.out_dir)
    manifest_path = _resolve(proj_root, args.manifest)

    maybe_clear_dir(out_dir, args.allow_overwrite)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    total_raw = 0
    total_nms = 0
    t0 = time.perf_counter()

    with manifest_path.open("w", newline="", encoding="utf-8") as fcsv:
        wr = csv.writer(fcsv)
        wr.writerow(["out_file", "src_image", "score", "note"])

        for idx, img_path in enumerate(files, start=1):
            img0 = imread_bgr(img_path, prefer_imageio=True)
            if img0 is None:
                continue

            kept, raw_n, nms_n = mine_one_image(
                img0, clf, hog, win_w, win_h,
                score_thr=args.score_thr,
                stride=args.stride,
                scale_factor=args.scale_factor,
                nms_iou_thr=args.nms_iou_thr,
                top_k=args.top_k
            )

            total_raw += raw_n
            total_nms += nms_n

            for j, (patch, s) in enumerate(kept, start=1):
                out_file = out_dir / f"{img_path.stem}_hn{j:02d}.jpg"
                imwrite_bgr(out_file, patch)
                total_saved += 1
                wr.writerow([str(out_file), str(img_path), f"{s:.6f}", f"thr={args.score_thr}"])

            if idx % 20 == 0:
                dt = (time.perf_counter() - t0) * 1000.0
                print(f"[{idx}] saved={total_saved}  raw_sum={total_raw}  nms_sum={total_nms}  elapsed(ms)={dt:.0f}")

    dt_all = (time.perf_counter() - t0) * 1000.0
    print("\n--- HNM ÖZET ---")
    print("İşlenen görüntü:", len(files))
    print("Kaydedilen hard-neg:", total_saved)
    print(f"Raw candidates (sum)={total_raw}  NMS candidates (sum)={total_nms}")
    print(f"Elapsed(ms)={dt_all:.0f}")
    print("Çıktı:", out_dir)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
