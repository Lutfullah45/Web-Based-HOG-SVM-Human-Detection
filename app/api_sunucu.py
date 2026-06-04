from pathlib import Path
import os
import time
import numpy as np
import cv2
import joblib
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from skimage.feature import hog as draw_hog
from skimage import exposure

PROJ_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv('INRIA_MODEL_PATH', PROJ_ROOT / 'baseline_work' / 'hog_svm_linear.joblib'))
if not MODEL_PATH.exists():
    raise FileNotFoundError(f'Model bulunamadı: {MODEL_PATH}. Önce scripts/train_hog_svm.py ile modeli üretin.')

WIN_W, WIN_H = 64, 128
hog = cv2.HOGDescriptor(
    _winSize=(WIN_W, WIN_H),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9
)

def hog_feat(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return hog.compute(gray).reshape(-1) #type: ignore

def nms(boxes, scores, iou_thr=0.4):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return keep

def detect(img0, clf, score_thr=0.5, iou_thr=0.4, stride=8, scale_factor=1.25, top_k=30):
    boxes, scores = [], []
    scale = 1.0
    img = img0.copy()

    while img.shape[1] >= WIN_W and img.shape[0] >= WIN_H:
        H, W = img.shape[:2]
        for y in range(0, H - WIN_H + 1, stride):
            for x in range(0, W - WIN_W + 1, stride):
                patch = img[y:y+WIN_H, x:x+WIN_W]
                f = hog_feat(patch)
                s = float(clf.decision_function([f])[0])
                if s >= score_thr:
                    xmin = int(x * scale)
                    ymin = int(y * scale)
                    xmax = int((x + WIN_W) * scale)
                    ymax = int((y + WIN_H) * scale)
                    boxes.append((xmin, ymin, xmax, ymax))
                    scores.append(s)

        scale *= scale_factor
        new_w = int(img0.shape[1] / scale)
        new_h = int(img0.shape[0] / scale)
        if new_w < WIN_W or new_h < WIN_H:
            break
        img = cv2.resize(img0, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    keep = nms(boxes, scores, iou_thr=iou_thr)
    kept = sorted([(boxes[i], scores[i]) for i in keep], key=lambda t: t[1], reverse=True)[:top_k]

    out = []
    for (xmin, ymin, xmax, ymax), s in kept:
        out.append({
            "xmin": int(xmin), "ymin": int(ymin),
            "xmax": int(xmax), "ymax": int(ymax),
            "score": float(s)
        })
    return out, len(boxes), len(keep)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

clf = joblib.load(MODEL_PATH)

@app.post("/detect")
async def detect_endpoint(file: UploadFile = File(...)):
    t0 = time.time()
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Görüntü okunamadı."}, status_code=400)

    # 1. Normal Nesne Tespiti (Arka planda çalışır)
    dets, raw_n, nms_n = detect(
        img, clf,
        score_thr=0.5,
        iou_thr=0.4,
        top_k=10
    )
    
    # 2. HOG Görselleştirme Şovu (Jüri için eklendi)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Hızlı çizim için görüntüyü biraz küçültelim
    if max(gray.shape) > 800:
        scale = 800 / max(gray.shape)
        gray = cv2.resize(gray, (int(gray.shape[1]*scale), int(gray.shape[0]*scale)))
        
    _, hog_img = draw_hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    hog_img_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 10))
    hog_8bit = (hog_img_rescaled * 255).astype(np.uint8)
    
    # HOG Görüntüsünü Web için Şifrele (Base64)
    _, buffer = cv2.imencode('.png', hog_8bit)
    hog_b64 = base64.b64encode(buffer).decode('utf-8')

    dt_ms = (time.time() - t0) * 1000.0

    return {
        "filename": file.filename,
        "raw_candidates": int(raw_n),
        "nms_candidates": int(nms_n),
        "detections": dets,
        "latency_ms": float(round(dt_ms, 2)),
        "hog_image": f"data:image/png;base64,{hog_b64}" # Resmi frontend'e yolluyoruz
    }