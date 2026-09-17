# Eye Disease Detection Backend

4-class FastAPI inference API for the React/Vite frontend. Loads the saved
checkpoint `model/efficientnetv2_b0_4class_best.keras` once at startup. The
model is not retrained or modified. Verified test accuracy on the v2 fundus
split remains **84.94%**.

Classes (API `class_id` only; the UI shows disease names):

- `0` Normal
- `1` Cataract
- `2` Diabetic Retinopathy
- `3` Glaucoma

Inference preprocessing is unchanged and matches the verified v2 notebooks:
PIL RGB → resize 224×224 BILINEAR → float32 pixels in 0–255. There is no
external `/255` and no augmentation. Grad-CAM uses `top_activation`.
Successful predictions are saved to MySQL table `prediction_history`.

## Camera / photo uploads

`app/image_intake/` runs **before** the existing inference pipeline. It accepts
JPG, JPEG, PNG, WEBP, BMP, TIFF, and HEIC, corrects EXIF orientation, converts
to RGB, checks brightness/blur, and crops a retinal region when a phone photo
contains extra background.

The trained model is validated on **retinal fundus images**. Camera-photo
support is an input-validation and preprocessing feature. It is **not**
evidence that the model was trained on arbitrary smartphone photographs of
the external eye. Uploads that are not a usable fundus image are rejected
with HTTP 400 and the message `Please upload a clear image`. They are not
classified as Normal, Cataract, Diabetic Retinopathy, or Glaucoma.

Verified fundus JPEGs that need no orientation fix or crop are passed to
`predict()` / `preprocess()` as the original file bytes.

## Endpoints

- `GET /health` — model + MySQL status
- `POST /predict` — multipart field name: `file`
- `POST /predict-with-gradcam` — same `file` field; prediction plus Grad-CAM images
- `GET /history` — newest-first prediction rows (`page`, `page_size`, `disease` or `class_id`, `q`)
- `GET /history/{id}` — stored prediction detail, including Grad-CAM images when saved
- `GET /dashboard` — aggregate counts and class distribution

Configure `backend/.env` from `.env.example`. Never put MySQL credentials in
frontend `VITE_*` variables.

Default XAMPP MySQL:

```
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=
MYSQL_DATABASE=eye_disease_detection
```

CORS allows the Vite frontend origins:

- http://localhost:5173
- http://127.0.0.1:5173

## Start

```powershell
cd "D:\Practice Projects\Disease Detection\backend"
& "D:\Practice Projects\Disease Detection\.venv311\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Swagger: http://127.0.0.1:8000/docs

## Tests

```powershell
cd "D:\Practice Projects\Disease Detection\backend"
& "D:\Practice Projects\Disease Detection\.venv311\Scripts\python.exe" test_image_intake.py
& "D:\Practice Projects\Disease Detection\.venv311\Scripts\python.exe" test_api.py
```

## Layout

- `app/main.py` — FastAPI routes
- `app/image_intake/` — format, decode, quality, fundus-suitability, ROI, workflow
- `app/model_service.py` — load, verified preprocess, predict
- `app/gradcam.py` — `top_activation` heatmaps
- `app/db.py` — MySQL `prediction_history` store
- `test_support/image_cases.py` — synthetic camera / reject fixtures
- `requirements.txt` — API dependencies

The previous 5-class product server is kept as `legacy_v1_app.py` (history,
dashboard, `/explain`). Do not start it for the current 4-class model.
