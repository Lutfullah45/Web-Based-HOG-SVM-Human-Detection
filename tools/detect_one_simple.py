"""
tools/detect_one_simple.py
Tek görüntü üzerinde hızlı doğrulama: kaba örnek pencerelerden HOG+SVM skoru üretir.

Bu sürüm, junction/symlink gerektirmeyen yeni dizin yapısına uygundur:
- proj_root: baseline_work ve model dosyasının olduğu kök
- data_root: ham INRIA verisinin olduğu kök (Train/Test burada)

Örnek:
python .\tools\detect_one_simple.py --prefer_imageio
python .\tools\detect_one_simple.py --image "Test/pos/crop001501.png" --prefer_imageio
"""

from __future__ import annotations

import argparse
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
        return img[..., ::-1].copy()  # RGB -> BGR
    return cv2.imread(str(path))


def pick_first_image(folder: Path):
    for ext in IMG_EXTS:
        pass
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            return p
    return None


def build_hog(win_w: int, win_h: int):
    return cv2.HOGDescriptor(
        _winSize=(win_w, win_h),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )


def hog_feat(hog, img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return hog.compute(gray).reshape(-1)  # type: ignore


def main():
    script_dir = Path(__file__).resolve().parent
    default_proj_root = script_dir.parent
    default_data_root = default_proj_root / "data" / "INRIA_raw"

    ap = argparse.ArgumentParser()
    ap.add_argument("--proj_root", type=str, default=str(default_proj_root))
    ap.add_argument("--data_root", type=str, default=str(default_data_root))
    ap.add_argument("--model_path", type=str, default="baseline_work/hog_svm_linear.joblib")
    ap.add_argument("--image", type=str, default="", help="data_root'e göre göreli yol (örn: Test/pos/xxx.png) veya tam yol")
    ap.add_argument("--prefer_imageio", action="store_true")

    ap.add_argument("--win_w", type=int, default=64)
    ap.add_argument("--win_h", type=int, default=128)
    ap.add_argument("--step", type=int, default=64)
    ap.add_argument("--score_thr", type=float, default=0.0, help="kaba eşik (debug)")
    args = ap.parse_args()

    proj_root = Path(args.proj_root)
    data_root = Path(args.data_root)
    model_path = proj_root / args.model_path

    clf = joblib.load(model_path)

    # Görüntü seçimi
    if args.image:
        ip = Path(args.image)
        img_path = ip if ip.is_absolute() else (data_root / ip)
    else:
        img_path = pick_first_image(data_root / "Test" / "pos")
        if img_path is None:
            raise RuntimeError("Test/pos içinde görüntü bulunamadı: " + str(data_root / "Test" / "pos"))

    img = imread_bgr(img_path, prefer_imageio=args.prefer_imageio)
    if img is None:
        raise RuntimeError("Görüntü okunamadı: " + str(img_path))

    hog = build_hog(args.win_w, args.win_h)

    H, W = img.shape[:2]
    candidates = []
    for y in range(0, max(1, H - args.win_h), args.step):
        for x in range(0, max(1, W - args.win_w), args.step):
            patch = img[y:y+args.win_h, x:x+args.win_w]
            if patch.shape[0] != args.win_h or patch.shape[1] != args.win_w:
                continue
            f = hog_feat(hog, patch)
            score = float(clf.decision_function([f])[0])
            if score >= args.score_thr:
                candidates.append((x, y, x + args.win_w, y + args.win_h, score))

    candidates.sort(key=lambda t: t[4], reverse=True)
    top = candidates[:5]

    print("PROJ_ROOT:", proj_root)
    print("DATA_ROOT:", data_root)
    print("Görüntü:", img_path)
    print("Bulunan aday sayısı:", len(candidates))
    print("Top-5 (xmin,ymin,xmax,ymax,score):")
    for t in top:
        print(t)


if __name__ == "__main__":
    main()
