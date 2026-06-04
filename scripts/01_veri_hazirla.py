import argparse
import csv
import random
import re
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np

# PASCAL-format bbox satırı:
# Bounding box for object ... : (Xmin, Ymin) - (Xmax, Ymax) : (250, 151) - (299, 294)
BBOX_RE = re.compile(r"Bounding box.*?:\s*\((\d+),\s*(\d+)\)\s*-\s*\((\d+),\s*(\d+)\)")

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def imread_bgr(path: Path):
    """Unicode-safe okuma. imageio genelde RGB döndürür; burada BGR'a çevrilir."""
    img = iio.imread(str(path))
    if img is None:
        return None

    # Gri görüntü -> 3 kanal
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    # RGBA -> RGB
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]

    # dtype güvenliği (nadiren uint16/fp gelebilir)
    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating):
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        elif img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

    # RGB -> BGR
    return img[..., ::-1].copy()


def imwrite_bgr(path: Path, img_bgr: np.ndarray):
    """Unicode-safe yazma. BGR -> RGB çevirip imageio ile yazar."""
    img_rgb = img_bgr[..., ::-1]
    iio.imwrite(str(path), img_rgb)


def parse_bboxes(ann_text: str):
    bboxes = []
    for m in BBOX_RE.finditer(ann_text):
        xmin, ymin, xmax, ymax = map(int, m.groups())
        if xmax > xmin and ymax > ymin:
            bboxes.append((xmin, ymin, xmax, ymax))
    return bboxes


def clip_box(box, w, h):
    xmin, ymin, xmax, ymax = box
    xmin = max(0, min(xmin, w - 1))
    ymin = max(0, min(ymin, h - 1))
    xmax = max(0, min(xmax, w))
    ymax = max(0, min(ymax, h))
    if xmax <= xmin or ymax <= ymin:
        return None
    return (xmin, ymin, xmax, ymax)


def pad_box(box, w, h, pad_frac=0.0):
    if pad_frac <= 0:
        return clip_box(box, w, h)
    xmin, ymin, xmax, ymax = box
    bw = xmax - xmin
    bh = ymax - ymin
    px = int(round(bw * pad_frac))
    py = int(round(bh * pad_frac))
    return clip_box((xmin - px, ymin - py, xmax + px, ymax + py), w, h)


def crop_resize(img_bgr, box, out_w, out_h, interp=cv2.INTER_LINEAR):
    xmin, ymin, xmax, ymax = box
    crop = img_bgr[ymin:ymax, xmin:xmax]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (out_w, out_h), interpolation=interp)


def grad_energy(patch_bgr):
    """Basit gradyan enerjisi: mean(|Sobel_x| + |Sobel_y|)."""
    g = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.abs(sx) + np.abs(sy)))


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


def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def maybe_clear_dir(d: Path, allow_overwrite: bool):
    d.mkdir(parents=True, exist_ok=True)
    if allow_overwrite:
        for f in d.glob("*"):
            if f.is_file():
                f.unlink()


def main():
    ap = argparse.ArgumentParser(
        description="INRIA Person: HOG+SVM için pozitif/negatif örnek hazırlama (imageio I/O, üretim sürümü)."
    )
    ap.add_argument("--proj_root", type=str, default=str(Path(__file__).resolve().parents[1]), help="Proje kökü (baseline_work burada)")
    ap.add_argument("--root", type=str, default="data/INRIA_raw", help="Ham veri kökü (Train/Test burada)")
    ap.add_argument("--out", type=str, default="baseline_work/prepared", help="Çıktı kök dizini")
    ap.add_argument("--win_w", type=int, default=64)
    ap.add_argument("--win_h", type=int, default=128)

    ap.add_argument("--max_pos_images", type=int, default=-1, help="-1: tümü, aksi: ilk N")
    ap.add_argument("--max_neg_images", type=int, default=-1, help="-1: tümü, aksi: ilk N")
    ap.add_argument("--neg_patches_per_image", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--pad_frac", type=float, default=0.05, help="Pozitif bbox padding oranı")
    ap.add_argument("--min_box_area", type=int, default=20 * 40, help="Çok küçük bbox eleme için min alan")
    ap.add_argument("--allow_overwrite", action="store_true", help="Çıktı klasörlerini temizleyip yeniden üret")

    ap.add_argument("--neg_grad_min", type=float, default=0.0,
                    help=">0 ise negatif yamada min gradyan enerjisi (düz yamaları elemek için)")
    ap.add_argument("--add_bg_negs_from_pos", action="store_true",
                    help="Train/pos görüntülerinden GT dışı arka plan negatifleri de üret")
    ap.add_argument("--bg_negs_per_pos_image", type=int, default=2)
    ap.add_argument("--bg_iou_max", type=float, default=0.05)

    args = ap.parse_args()
    random.seed(args.seed)

    proj_root = Path(args.proj_root)
    root = Path(args.root)
    if not root.is_absolute():
        root = proj_root / root
    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = proj_root / out_root
    out_pos = out_root / "pos"
    out_neg = out_root / "neg"
    manifest_path = out_root / "manifest.csv"

    maybe_clear_dir(out_pos, args.allow_overwrite)
    maybe_clear_dir(out_neg, args.allow_overwrite)
    out_root.mkdir(parents=True, exist_ok=True)

    train_pos_dir = root / "Train" / "pos"
    train_neg_dir = root / "Train" / "neg"
    train_ann_dir = root / "Train" / "annotations"

    pos_images = list_images(train_pos_dir)
    neg_images = list_images(train_neg_dir)

    if args.max_pos_images > 0:
        pos_images = pos_images[:args.max_pos_images]
    if args.max_neg_images > 0:
        neg_images = neg_images[:args.max_neg_images]

    pos_count = 0
    neg_count = 0

    with manifest_path.open("w", newline="", encoding="utf-8") as fcsv:
        wr = csv.writer(fcsv)
        wr.writerow(["type", "out_file", "src_image", "src_ann", "xmin", "ymin", "xmax", "ymax", "note"])

        # --- Pozitifler ---
        for img_path in pos_images:
            base = img_path.stem
            ann_path = train_ann_dir / f"{base}.txt"
            if not ann_path.exists():
                continue

            img = imread_bgr(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            ann_text = ann_path.read_text(encoding="utf-8", errors="ignore")
            bboxes = parse_bboxes(ann_text)

            obj_idx = 0
            for bb in bboxes:
                xmin, ymin, xmax, ymax = bb
                if (xmax - xmin) * (ymax - ymin) < args.min_box_area:
                    continue

                bb2 = pad_box(bb, w, h, pad_frac=args.pad_frac)
                if bb2 is None:
                    continue

                patch = crop_resize(img, bb2, args.win_w, args.win_h)
                if patch is None:
                    continue

                obj_idx += 1
                out_file = out_pos / f"{base}_obj{obj_idx:02d}.jpg"
                imwrite_bgr(out_file, patch)
                pos_count += 1
                wr.writerow(["pos", str(out_file), str(img_path), str(ann_path),
                             bb2[0], bb2[1], bb2[2], bb2[3], f"pad={args.pad_frac}"])

            # --- Opsiyonel: pos görüntüden arka plan negatifleri ---
            if args.add_bg_negs_from_pos and obj_idx > 0:
                gt = []
                for bb in bboxes:
                    bbx = clip_box(bb, w, h)
                    if bbx is not None:
                        gt.append(bbx)

                trials, made = 0, 0
                max_trials = 200
                while made < args.bg_negs_per_pos_image and trials < max_trials:
                    trials += 1
                    if w < args.win_w or h < args.win_h:
                        break
                    x = random.randint(0, w - args.win_w)
                    y = random.randint(0, h - args.win_h)
                    cand = (x, y, x + args.win_w, y + args.win_h)

                    if gt and max(iou(cand, g) for g in gt) > args.bg_iou_max:
                        continue

                    patch = img[y:y + args.win_h, x:x + args.win_w]
                    if patch.size == 0:
                        continue
                    if args.neg_grad_min > 0.0 and grad_energy(patch) < args.neg_grad_min:
                        continue

                    out_file = out_neg / f"{base}_bgneg{made + 1:02d}.jpg"
                    imwrite_bgr(out_file, patch)
                    neg_count += 1
                    made += 1
                    wr.writerow(["neg_bg", str(out_file), str(img_path), str(ann_path),
                                 cand[0], cand[1], cand[2], cand[3], f"bg_iou_max={args.bg_iou_max}"])

        # --- Negatifler (Train/neg) ---
        for img_path in neg_images:
            img = imread_bgr(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            if w < args.win_w or h < args.win_h:
                continue

            for k in range(args.neg_patches_per_image):
                x = random.randint(0, w - args.win_w)
                y = random.randint(0, h - args.win_h)
                patch = img[y:y + args.win_h, x:x + args.win_w]
                if patch.size == 0:
                    continue
                if args.neg_grad_min > 0.0 and grad_energy(patch) < args.neg_grad_min:
                    continue

                out_file = out_neg / f"{img_path.stem}_neg{k + 1:02d}.jpg"
                imwrite_bgr(out_file, patch)
                neg_count += 1
                wr.writerow(["neg", str(out_file), str(img_path), "", x, y, x + args.win_w, y + args.win_h,
                             f"grad_min={args.neg_grad_min}"])

    print("Hazırlık tamamlandı.")
    print("Pozitif örnek sayısı:", pos_count)
    print("Negatif örnek sayısı:", neg_count)
    print("Çıktı klasörleri:")
    print("  ", out_pos)
    print("  ", out_neg)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
