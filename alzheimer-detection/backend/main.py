"""FastAPI backend.  Run:  uvicorn main:app --reload --port 8000"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from pipeline import CLASS_NAMES, DEVICE, MAX_FILE_MB, InvalidImageError, load_model, predict

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = load_model()  # load once at startup
    yield
    state.clear()


app = FastAPI(title="Alzheimer's MRI Detection API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "classes": CLASS_NAMES}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_FILE_MB} MB.")
    try:
        result = predict(state["model"], data, file.filename)
    except InvalidImageError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # unexpected model/runtime error
        raise HTTPException(500, f"Inference failed: {e}")
    return {
        "filename": file.filename,
        "prediction": result.label,
        "confidence": result.confidence,
        "probabilities": result.probabilities,
    }
