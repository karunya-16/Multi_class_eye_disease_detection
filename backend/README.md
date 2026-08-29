# Eye Disease Detection Backend

Product FastAPI app for the React frontend. Inference uses the saved
`model/efficientnetv2_finetuned_best.keras` checkpoint. The model is not retrained.

Endpoints:

- `GET /`
- `GET /health`
- `POST /predict` — multipart field name: `file` (JPG / JPEG / PNG)
- `POST /explain` — same `file` field; prediction plus Grad-CAM images when available
- `POST /history` — save a completed successful analysis (`request_id` prevents duplicate rows)
- `GET /analytics/dashboard` — aggregate MySQL statistics (`predicted_class`/`predicted_label`, `start_date`, `end_date`)
- `GET /history` — newest-first paginated history (`page`, `page_size`, `predicted_class` or `predicted_label`, `q`, `start_date`, `end_date`)
- `GET /history/{id}` — stored prediction detail
- `GET /history/{id}/image/{uploaded|original|heatmap|overlay}`

Completed results are stored in the MySQL table `analysis_results`. Uploaded and Grad-CAM
images are saved as files under `backend/data/history_files/`; MySQL stores those file
paths, not image BLOBs. Configure `backend/.env` from `.env.example`. Never put MySQL
credentials in frontend `VITE_*` variables.

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
& "D:\Practice Projects\Disease Detection\.venv311\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Swagger: http://127.0.0.1:8000/docs
