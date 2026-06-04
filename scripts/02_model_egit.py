"""
train_hog_svm.py
INRIA Person için HOG özellikleri + Lineer SVM (baseline) eğitimi.

Bu sürüm, train_hog_svm_mini.py dosyasının geliştirilmiş halidir:
- Windows/Unicode yol uyumu için imageio ile okuma
- Negatif örnek aşırı fazlaysa subsampling (max_neg)
- class_weight='balanced' ile sınıf dengesizliğini azaltma (opsiyon)
- C için küçük bir grid ile en iyi modeli seçme
- Basit cache (HOG özelliklerini .npz kaydetme) (opsiyon)
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import imageio.v3 as iio
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def imread_bgr(path: Path):
    """Unicode-safe okuma. imageio genelde RGB döndürür; burada BGR'a çevrilir."""
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


def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def build_hog(win_w: int, win_h: int, cell: int = 8, block_cells: int = 2, nbins: int = 9):
    block = (cell * block_cells, cell * block_cells)
    return cv2.HOGDescriptor(
        _winSize=(win_w, win_h),
        _blockSize=block,
        _blockStride=(cell, cell),
        _cellSize=(cell, cell),
        _nbins=nbins
    )


def hog_feat(hog, img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    feat = hog.compute(gray)
    return feat.reshape(-1)


def extract_features(
    hog,
    files,
    label: int,
    win_w: int,
    win_h: int,
    cache_path: Path | None = None,
    verbose_every: int = 500
):
    if cache_path is not None and cache_path.exists():
        data = np.load(str(cache_path), allow_pickle=True)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int32)
        return X, y

    X_list, y_list = [], []
    for i, p in enumerate(files, start=1):
        img = imread_bgr(p)
        if img is None:
            continue
        if img.shape[1] != win_w or img.shape[0] != win_h:
            img = cv2.resize(img, (win_w, win_h), interpolation=cv2.INTER_LINEAR)

        X_list.append(hog_feat(hog, img))
        y_list.append(label)

        if verbose_every and (i % verbose_every == 0):
            print(f"  okunan: {i}/{len(files)} (label={label})")

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int32)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(cache_path), X=X, y=y)

    return X, y


def main():
    ap = argparse.ArgumentParser(description="HOG + Lineer SVM eğitimi (INRIA baseline).")
    ap.add_argument("--proj_root", type=str, default=str(Path(__file__).resolve().parents[1]), help="Proje kökü (baseline_work burada)")
    ap.add_argument("--pos_dir", type=str, default="baseline_work/prepared/pos")
    ap.add_argument("--neg_dir", type=str, default="baseline_work/prepared/neg")
    ap.add_argument("--out_model", type=str, default="baseline_work/hog_svm_linear.joblib")
    ap.add_argument("--out_meta", type=str, default="baseline_work/hog_svm_linear_meta.json")

    ap.add_argument("--win_w", type=int, default=64)
    ap.add_argument("--win_h", type=int, default=128)
    ap.add_argument("--cell", type=int, default=8)
    ap.add_argument("--block_cells", type=int, default=2)
    ap.add_argument("--nbins", type=int, default=9)

    ap.add_argument("--max_pos", type=int, default=-1, help="-1 tümü, aksi halde ilk N")
    ap.add_argument("--max_neg", type=int, default=-1, help="-1 tümü, aksi halde rastgele N (negatif subsample)")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--class_weight", type=str, default="balanced", choices=["balanced", "none"])
    ap.add_argument("--c_list", type=float, nargs="*", default=[0.1, 1.0, 5.0], help="C değerleri listesi")
    ap.add_argument("--max_iter", type=int, default=20000)

    ap.add_argument("--cache_dir", type=str, default="", help="örn: baseline_work/cache (boşsa cache kapalı)")
    ap.add_argument("--verbose_every", type=int, default=500)

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    proj_root = Path(args.proj_root)
    pos_dir = Path(args.pos_dir)
    neg_dir = Path(args.neg_dir)
    if not pos_dir.is_absolute():
        pos_dir = proj_root / pos_dir
    if not neg_dir.is_absolute():
        neg_dir = proj_root / neg_dir

    pos_files = list_images(pos_dir)
    neg_files = list_images(neg_dir)

    if args.max_pos > 0:
        pos_files = pos_files[:args.max_pos]

    if args.max_neg > 0:
        # negatifleri rastgele seçmek daha sağlıklı (sıralı seçim bias yapabilir)
        if len(neg_files) > args.max_neg:
            neg_files = random.sample(neg_files, args.max_neg)

    if len(pos_files) == 0 or len(neg_files) == 0:
        raise RuntimeError("Pozitif veya negatif dosya listesi boş. Klasör yollarını kontrol edin.")

    print("Pozitif dosya:", len(pos_files))
    print("Negatif dosya:", len(neg_files))

    hog = build_hog(args.win_w, args.win_h, cell=args.cell, block_cells=args.block_cells, nbins=args.nbins)

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir is not None and (not cache_dir.is_absolute()):
        cache_dir = proj_root / cache_dir
    pos_cache = (cache_dir / "pos.npz") if cache_dir else None
    neg_cache = (cache_dir / "neg.npz") if cache_dir else None

    print("Öznitelik çıkarımı (pos)...")
    Xp, yp = extract_features(
        hog, pos_files, 1, args.win_w, args.win_h,
        cache_path=pos_cache, verbose_every=args.verbose_every
    )
    print("Öznitelik çıkarımı (neg)...")
    Xn, yn = extract_features(
        hog, neg_files, 0, args.win_w, args.win_h,
        cache_path=neg_cache, verbose_every=args.verbose_every
    )

    if len(Xp) == 0 or len(Xn) == 0:
        raise RuntimeError("Öznitelik çıkarımı sonucunda X boş kaldı. Görüntü okuma/yol sorunlarını kontrol edin.")

    X = np.vstack([Xp, Xn])
    y = np.concatenate([yp, yn])

    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    cw = None if args.class_weight == "none" else "balanced"

    best = None
    best_f1 = -1.0

    for C in args.c_list:
        clf = make_pipeline(
            StandardScaler(with_mean=True),
            LinearSVC(C=C, class_weight=cw, max_iter=args.max_iter)
        )
        clf.fit(Xtr, ytr)
        yhat = clf.predict(Xva)
        f1 = f1_score(yva, yhat, average="binary")
        print(f"[C={C}] val F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best = (C, clf, yhat)

    C_best, clf_best, yhat_best = best #type:ignore

    print("\n--- VALIDATION RAPORU (en iyi C) ---")
    print("C_best =", C_best)
    print(classification_report(yva, yhat_best, digits=4))
    cm = confusion_matrix(yva, yhat_best)
    print("Confusion matrix [[TN FP],[FN TP]]:\n", cm)

    out_model = Path(args.out_model)
    if not out_model.is_absolute():
        out_model = proj_root / out_model
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf_best, out_model)
    print("Model kaydedildi:", out_model)

    meta = {
        "pos_dir": str(pos_dir),
        "neg_dir": str(neg_dir),
        "n_pos": int(len(pos_files)),
        "n_neg": int(len(neg_files)),
        "hog": {
            "win_w": args.win_w, "win_h": args.win_h,
            "cell": args.cell, "block_cells": args.block_cells, "nbins": args.nbins
        },
        "svm": {
            "C_best": float(C_best),
            "class_weight": args.class_weight,
            "max_iter": args.max_iter
        },
        "val": {
            "f1_best": float(best_f1),
            "test_size": float(args.test_size),
            "seed": int(args.seed)
        }
    }
    out_meta = Path(args.out_meta)
    if not out_meta.is_absolute():
        out_meta = proj_root / out_meta
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Meta kaydedildi:", out_meta)


if __name__ == "__main__":
    main()
