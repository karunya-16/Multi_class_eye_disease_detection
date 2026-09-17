# Eye Disease Detection Frontend

Vite + React UI for the 4-class FastAPI backend at `http://127.0.0.1:8000`.

- Analysis: camera or file upload → `POST /predict`, `POST /predict-with-gradcam`
- Dashboard: `GET /dashboard`
- History: `GET /history`, `GET /history/{id}`

Predicted labels in the UI are disease names: Normal, Cataract, Diabetic
Retinopathy, Glaucoma. The screening model is validated on retinal fundus
images (84.94% test accuracy). Camera/photo support is input validation and
preprocessing only. Unsuitable images show **Please upload a clear image**.

## Setup

```powershell
cd "D:\Practice Projects\Disease Detection\frontend"
npm install
copy .env.example .env
```

`.env` should contain:

```
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## Start

Start the backend first, then:

```powershell
cd "D:\Practice Projects\Disease Detection\frontend"
npm run dev
```

Open http://localhost:5173
