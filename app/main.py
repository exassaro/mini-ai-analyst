"""
Mini AI Analyst as a Service (AaaS)
====================================
Main FastAPI application entry point.
Initializes the app, mounts routers, serves the frontend,
and ensures storage directories exist on startup.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.upload import router as upload_router
from app.api.profile import router as profile_router
from app.api.train import router as train_router
from app.api.predict import router as predict_router
from app.api.summary import router as summary_router
from app.core.config import settings

# ── Application factory ─────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Upload a CSV → profile it → train a model → get predictions.",
)

# ── CORS (allow the frontend to call the API) ───────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────
app.include_router(upload_router, prefix="/upload", tags=["Upload"])
app.include_router(profile_router, prefix="/profile", tags=["Profile"])
app.include_router(train_router, prefix="/train", tags=["Train"])
app.include_router(predict_router, prefix="/predict", tags=["Predict"])
app.include_router(summary_router, prefix="/summary", tags=["Summary"])


# ── Startup event ────────────────────────────────────────────────────
@app.on_event("startup")
def _create_storage_dirs() -> None:
    """Ensure the upload and model storage directories exist."""
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.MODEL_DIR, exist_ok=True)


# ── Serve the frontend (static files) ───────────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


# ── Health-check (useful for quick smoke tests) ─────────────────────
@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "app": settings.APP_NAME, "version": settings.APP_VERSION}
