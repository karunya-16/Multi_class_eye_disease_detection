# Eye Disease Detection Frontend

Vite + React UI that sends a fundus image to the FastAPI backend at
`POST /predict` (multipart field: `file`).

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
