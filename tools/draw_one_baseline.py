"""
tools/draw_one_baseline.py (ultra-safe save)
- Okuma: imageio (opsiyonel) / cv2
- Yazma: cv2.imencode + Python write_bytes (Unicode/Drive uyumlu)

Bu sürüm, cv2.imwrite ve bazı imageio backend'lerinde görülebilen yol/FS sorunlarına karşı
çıktıyı doğrudan byte olarak yazar.

Kullanım:
python .\tools\draw_one_baseline.py --image "Test/pos/crop001659.png" --score_thr 0.5 --top_k_draw 5 --prefer_imageio
"""

from __future__ import annotations

import argparse
import json
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
        return img[..., ::-1].copy()
    return cv2.imread(str(path))


def pick_first_image(folder: Path):
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            return p
    return None


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


def hog_feat(hog, img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return hog.compute(gray).reshape(-1)  # type: ignore


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


def write_jpg_bytes(path: Path, img_bgr: np.ndarray, quality: int = 95):
    """cv2.imencode + Path.write_bytes ile güvenli JPG yaz."""
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buf = cv2.imencode(".jpg", img_bgr, params)
    if not ok:
        raise RuntimeError("cv2.imencode('.jpg', ...) başarısız.")
    path.write_bytes(buf.tobytes())


def main():
    script_dir = Path(__file__).resolve().parent
    default_proj_root = script_dir.parent
    default_data_root = default_proj_root / "data" / "INRIA_raw"

    ap = argparse.ArgumentParser()
    ap.add_argument("--proj_root", type=str, default=str(default_proj_root))
    ap.add_argument("--data_root", type=str, default=str(default_data_root))
    ap.add_argument("--model_path", type=str, default="baseline_work/hog_svm_linear.joblib")
    ap.add_argument("--meta_path", type=str, default="baseline_work/hog_svm_linear_meta.json")
    ap.add_argument("--out_dir", type=str, default="baseline_work/out")

    ap.add_argument("--image", type=str, default="")
    ap.add_argument("--prefer_imageio", action="store_true")

    ap.add_argument("--score_thr", type=float, default=2.0)
    ap.add_argument("--nms_iou_thr", type=float, default=0.3)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--scale_factor", type=float, default=1.25)

    ap.add_argument("--top_k_draw", type=int, default=5)
    ap.add_argument("--jpg_quality", type=int, default=95)
    args = ap.parse_args()

    proj_root = Path(args.proj_root)
    data_root = Path(args.data_root)
    out_dir = proj_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    clf = joblib.load(proj_root / args.model_path)
    hog, win_w, win_h = build_hog_from_meta(proj_root / args.meta_path, fallback_win=(64, 128))

    if args.image:
        ip = Path(args.image)
        img_path = ip if ip.is_absolute() else (data_root / ip)
    else:
        img_path = pick_first_image(data_root / "Test" / "pos")
        if img_path is None:
            raise RuntimeError("Test/pos içinde görüntü bulunamadı: " + str(data_root / "Test" / "pos"))

    img0 = imread_bgr(img_path, prefer_imageio=args.prefer_imageio)
    if img0 is None:
        raise RuntimeError("Görüntü okunamadı: " + str(img_path))

    boxes, scores = [], []
    scale = 1.0
    img = img0

    while img.shape[1] >= win_w and img.shape[0] >= win_h:
        H, W = img.shape[:2]
        for y in range(0, H - win_h + 1, args.stride):
            for x in range(0, W - win_w + 1, args.stride):
                patch = img[y:y + win_h, x:x + win_w]
                s = float(clf.decision_function([hog_feat(hog, patch)])[0])
                if s >= args.score_thr:
                    xmin = int(x * scale)
                    ymin = int(y * scale)
                    xmax = int((x + win_w) * scale)
                    ymax = int((y + win_h) * scale)
                    boxes.append((xmin, ymin, xmax, ymax))
                    scores.append(s)

        scale *= args.scale_factor
        new_w = int(img0.shape[1] / scale)
        new_h = int(img0.shape[0] / scale)
        if new_w < win_w or new_h < win_h:
            break
        img = cv2.resize(img0, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    keep = nms(boxes, scores, iou_thr=args.nms_iou_thr)
    kept = sorted([(boxes[i], scores[i]) for i in keep], key=lambda t: t[1], reverse=True)

    K = max(0, args.top_k_draw)
    vis = img0.copy()
    for (xmin, ymin, xmax, ymax), s in kept[:K]:
        cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(vis, f"{float(s):.2f}", (xmin, max(0, ymin - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    out_path = out_dir / f"{img_path.stem}_det.jpg"
    write_jpg_bytes(out_path, vis, quality=args.jpg_quality)

    # Yazıldı mı?
    ok = out_path.exists()
    size = out_path.stat().st_size if ok else 0

    print("PROJ_ROOT:", proj_root)
    print("DATA_ROOT:", data_root)
    print("Görüntü:", img_path)
    print("Ham aday:", len(boxes))
    print("NMS sonrası:", len(keep))
    print("Çizilen kutu:", min(K, len(kept)))
    print("Kaydedildi:", out_path)
    print("Dosya var mı?:", ok, "Boyut(bytes):", size)


if __name__ == "__main__":
    main()
