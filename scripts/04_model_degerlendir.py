"""
scripts/eval_inria_detection.py
INRIA Person (Test/pos) üzerinde HOG+SVM deteksiyon değerlendirmesi (IoU eşleşmeli P/R/F1).

Bu sürüm, ham veri setinin `data/INRIA_raw` altında tutulduğu (junction/symlink yok) senaryoya uygundur.

- --proj_root: proje kökü (baseline_work burada)
- --data_root: veri kökü (Train/Test burada)

Örnek:
python .\scripts\eval_inria_detection.py --prefer_imageio --max_images 288 --score_thr 2.0 --top_k 3
"""

from __future__ import annotations

import argparse
import csv
import json
import re
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

BBOX_RE = re.compile(r"Bounding box.*?:\s*\((\d+),\s*(\d+)\)\s*-\s*\((\d+),\s*(\d+)\)")
IMG_RE = re.compile(r'Image filename\s*:\s*"(.*?)"')


def parse_ann(ann_path: Path):
    txt = ann_path.read_text(encoding="utf-8", errors="ignore")
    m = IMG_RE.search(txt)
    if not m:
        return None, []
    rel_img = m.group(1)
    bboxes = []
    for bb in BBOX_RE.finditer(txt):
        xmin, ymin, xmax, ymax = map(int, bb.groups())
        if xmax > xmin and ymax > ymin:
            bboxes.append((xmin, ymin, xmax, ymax))
    return rel_img, bboxes


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
    return hog, win_w, win_h, cell, block_cells, nbins


def hog_feat(hog, img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return hog.compute(gray).reshape(-1)  # type: ignore


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    xx1 = max(ax1, bx1)
    yy1 = max(ay1, by1)
    xx2 = min(ax2, bx2)
    yy2 = min(ay2, by2)
    w = max(0, xx2 - xx1)
    h = max(0, yy2 - yy1)
    inter = w * h
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def nms(boxes, scores, iou_thr: float):
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iouv = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iouv <= iou_thr)[0]
        order = order[inds + 1]
    return keep


def detect(img0, clf, hog, win_w, win_h, score_thr, nms_iou_thr, top_k, stride, scale_factor):
    boxes, scores = [], []
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

        scale *= scale_factor
        new_w = int(img0.shape[1] / scale)
        new_h = int(img0.shape[0] / scale)
        if new_w < win_w or new_h < win_h:
            break
        img = cv2.resize(img0, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    keep = nms(boxes, scores, iou_thr=nms_iou_thr)
    kept = sorted([(boxes[i], scores[i]) for i in keep], key=lambda t: t[1], reverse=True)
    if top_k > 0:
        kept = kept[:top_k]
    return kept, len(boxes), len(keep)


def match_counts(preds, gts, match_iou_thr: float):
    used = [False] * len(gts)
    tp = fp = 0
    for pb, _ in preds:
        best_iou = 0.0
        best_j = -1
        for j, gb in enumerate(gts):
            if used[j]:
                continue
            v = iou(pb, gb)
            if v > best_iou:
                best_iou = v
                best_j = j
        if best_iou >= match_iou_thr and best_j >= 0:
            used[best_j] = True
            tp += 1
        else:
            fp += 1
    fn = used.count(False)
    return tp, fp, fn


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
    ap.add_argument("--ann_dir", type=str, default="Test/annotations")

    ap.add_argument("--score_thr", type=float, default=2.0)
    ap.add_argument("--nms_iou_thr", type=float, default=0.3)
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--scale_factor", type=float, default=1.25)

    ap.add_argument("--match_iou_thr", type=float, default=0.50)
    ap.add_argument("--max_images", type=int, default=288)
    ap.add_argument("--prefer_imageio", action="store_true")
    ap.add_argument("--csv_out", type=str, default="")
    ap.add_argument("--verbose_every", type=int, default=30)
    args = ap.parse_args()

    proj_root = Path(args.proj_root)
    data_root = Path(args.data_root)

    model_path = _resolve(proj_root, args.model_path)
    meta_path = _resolve(proj_root, args.meta_path)

    clf = joblib.load(model_path)
    hog, win_w, win_h, cell, block_cells, nbins = build_hog_from_meta(meta_path, fallback_win=(64, 128))

    ann_dir = _resolve(data_root, args.ann_dir)
    anns = sorted([p for p in ann_dir.iterdir() if p.is_file()])

    total_tp = total_fp = total_fn = 0
    times_ms = []
    raw_list = []
    nms_list = []
    processed = 0
    skipped_img = 0

    csv_writer = None
    csv_file = None
    if args.csv_out:
        outp = _resolve(proj_root, args.csv_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        csv_file = outp.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["img", "tp", "fp", "fn", "raw", "nms", "lat_ms"])

    for ann in anns:
        rel_img, gts = parse_ann(ann)
        if rel_img is None:
            continue
        if not rel_img.replace("\\", "/").startswith("Test/pos/"):
            continue

        img_path = _resolve(data_root, rel_img)
        img = imread_bgr(img_path, prefer_imageio=True)
        if img is None:
            skipped_img += 1
            continue

        t0 = time.perf_counter()
        preds, raw_n, nms_n = detect(
            img, clf, hog, win_w, win_h,
            score_thr=args.score_thr,
            nms_iou_thr=args.nms_iou_thr,
            top_k=args.top_k,
            stride=args.stride,
            scale_factor=args.scale_factor
        )
        dt = (time.perf_counter() - t0) * 1000.0

        tp, fp, fn = match_counts(preds, gts, match_iou_thr=args.match_iou_thr)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        times_ms.append(dt)
        raw_list.append(raw_n)
        nms_list.append(nms_n)
        processed += 1

        if csv_writer is not None:
            csv_writer.writerow([str(img_path).replace("\\", "/"), tp, fp, fn, raw_n, nms_n, f"{dt:.2f}"])

        if args.verbose_every and (processed % args.verbose_every == 0):
            print(f"[{processed}] last latency(ms)={dt:.2f}  TP/FP/FN={tp}/{fp}/{fn}  raw/nms={raw_n}/{nms_n}")

        if processed >= args.max_images:
            break

    if csv_file is not None:
        csv_file.close()

    prec = total_tp / (total_tp + total_fp + 1e-9)
    rec = total_tp / (total_tp + total_fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)

    if times_ms:
        avg_ms = float(np.mean(times_ms))
        med_ms = float(np.median(times_ms))
        p95_ms = float(np.percentile(times_ms, 95))
    else:
        avg_ms = med_ms = p95_ms = 0.0

    print("\n--- ÖZET ---")
    print("Değerlendirilen görüntü:", processed)
    if skipped_img:
        print("Okunamayan görüntü:", skipped_img)
    print(f"TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    print(f"Latency(ms): avg={avg_ms:.2f}  median={med_ms:.2f}  p95={p95_ms:.2f}")
    if raw_list:
        print(f"Candidates: raw(avg)={np.mean(raw_list):.2f}  nms(avg)={np.mean(nms_list):.2f}")

    print("\n--- PARAMETRELER ---")
    print(f"PROJ_ROOT: {proj_root}")
    print(f"DATA_ROOT: {data_root}")
    print(f"HOG: win={win_w}x{win_h}, cell={cell}, block_cells={block_cells}, nbins={nbins}")
    print(f"Detect: score_thr={args.score_thr}, top_k={args.top_k}, stride={args.stride}, scale_factor={args.scale_factor}, nms_iou_thr={args.nms_iou_thr}")
    print(f"Eval: match_iou_thr={args.match_iou_thr}, max_images={args.max_images}")


if __name__ == "__main__":
    main()
