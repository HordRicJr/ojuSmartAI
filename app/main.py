import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.routes.vision import router as vision_router
from app.core.config import settings
from app.services.inference import load_models

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Gestionnaire de cycle de vie de l'application FastAPI.

    Charge tous les modeles IA une seule fois au demarrage via load_models()
    et les stocke dans app.state. Libere la memoire GPU au shutdown.
    """
    logger.info("Demarrage du service OjuSmart AI Engine...")
    models = load_models()

    application.state.device = models["device"]
    application.state.signature_model = models["signature_model"]
    application.state.signature_transform = models["signature_transform"]
    application.state.face_detector = models["face_detector"]
    application.state.emotion_pipeline = models["emotion_pipeline"]
    application.state.blip_processor = models["blip_processor"]
    application.state.blip_model = models["blip_model"]

    logger.info("Tous les modeles sont charges. Service pret.")

    try:
        yield
    finally:
        del application.state.signature_model
        del application.state.blip_model
        del application.state.emotion_pipeline

        device: torch.device = models["device"]
        if device.type == "cuda":
            torch.cuda.empty_cache()

        logger.info("Ressources liberees. Service arrete.")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Moteur d'inference IA stateless pour la plateforme OjuSmart. "
        "Ce service traite les images de signatures, de visages et d'environnements "
        "transmises par le backend Spring Boot et retourne des resultats en JSON. "
        "Aucune connexion a une base de donnees n'est etablie."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.include_router(vision_router)


@app.get(
    "/health",
    summary="Verification de l'etat du service",
    tags=["Monitoring"],
)
async def health_check() -> JSONResponse:
    """Verifie que le service est operationnel.

    Retourne statut, nom, version et dispositif de calcul actif (cpu/cuda/mps).
    Utilise par Spring Boot et les outils de monitoring (Kubernetes liveness probe).
    """
    device_name: str = str(app.state.device) if hasattr(app.state, "device") else "inconnu"
    return JSONResponse(
        content={
            "status": "healthy",
            "service": settings.app_name,
            "version": settings.app_version,
            "device": device_name,
        }
    )
