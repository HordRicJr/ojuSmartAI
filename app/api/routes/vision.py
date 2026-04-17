from fastapi import APIRouter, HTTPException, Request, UploadFile, status
from transformers import BlipForConditionalGeneration, BlipProcessor

from app.models.schemas import DescriptionResponse, EmbeddingResponse, EmotionResponse
from app.services.inference import VisionService

router = APIRouter(prefix="/ai/v1", tags=["Vision"])

_vision_service = VisionService()


@router.post(
    "/signatures/analyze",
    response_model=EmbeddingResponse,
    summary="Extraire l'embedding d'une image de signature",
    description=(
        "Recoit une image de signature en multipart/form-data, extrait un vecteur "
        "de 2048 caracteristiques via ResNet50 timm et le retourne en JSON."
    ),
)
def analyze_signature(request: Request, file: UploadFile) -> EmbeddingResponse:
    """Point d'entree pour l'extraction d'embedding de signature.

    Synchrone afin que FastAPI l'execute dans un threadpool (calculs PyTorch bloquants).

    Parametres
    ----------
    request : Request
        Acces a app.state (signature_model, signature_transform).
    file : UploadFile
        Image transmise par Spring Boot via multipart/form-data.

    Retourne
    --------
    EmbeddingResponse
        Vecteur de 2048 dimensions et sa taille.

    Leve
    ----
    HTTPException 400
        Image invalide ou format non supporte.
    HTTPException 500
        Erreur interne non anticipee.
    """
    try:
        image_bytes: bytes = file.file.read()
        pil_image = _vision_service.preprocess_signature(image_bytes)
        embedding = _vision_service.get_signature_embedding(
            pil_image,
            request.app.state.signature_model,
            request.app.state.signature_transform,
        )
        return EmbeddingResponse(embedding=embedding, dimension=len(embedding))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Une erreur interne s'est produite lors du traitement de l'image.",
        ) from exc


@router.post(
    "/emotions/detect",
    response_model=EmotionResponse,
    summary="Detecter l'emotion dans une image",
    description=(
        "Recoit une image en multipart/form-data et detecte l'emotion dominante "
        "via dima806/facial_emotions_image_detection (7 classes HF)."
    ),
)
def detect_emotion(request: Request, file: UploadFile) -> EmotionResponse:
    """Point d'entree pour la detection d'emotion.

    Synchrone afin que FastAPI l'execute dans un threadpool (pipeline HF bloquant).

    Parametres
    ----------
    request : Request
        Acces a app.state (emotion_pipeline).
    file : UploadFile
        Image transmise par Spring Boot via multipart/form-data.

    Retourne
    --------
    EmotionResponse
        Libelle d'emotion et score de confiance.

    Leve
    ----
    HTTPException 400
        Image invalide ou aucune emotion detectee.
    HTTPException 500
        Erreur interne non anticipee.
    """
    try:
        image_bytes: bytes = file.file.read()
        emotion, confidence = _vision_service.analyze_emotion(
            image_bytes,
            request.app.state.emotion_pipeline,
            request.app.state.face_detector,
        )
        return EmotionResponse(emotion=emotion, confidence=confidence)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Une erreur interne s'est produite lors du traitement de l'image.",
        ) from exc


@router.post(
    "/environment/describe",
    response_model=DescriptionResponse,
    summary="Generer une description textuelle de l'environnement",
    description=(
        "Recoit une image en multipart/form-data et genere une phrase descriptive "
        "via Salesforce/blip-image-captioning-base. "
        "Exemple de reponse : 'a wooden desk with a computer on top'."
    ),
)
def describe_environment(request: Request, file: UploadFile) -> DescriptionResponse:
    """Point d'entree pour la description visuelle de l'environnement.

    Synchrone afin que FastAPI l'execute dans un threadpool (generation BLIP bloquante).

    Parametres
    ----------
    request : Request
        Acces a app.state (blip_processor, blip_model).
    file : UploadFile
        Image transmise par Spring Boot via multipart/form-data.

    Retourne
    --------
    DescriptionResponse
        Phrase generee par BLIP.

    Leve
    ----
    HTTPException 400
        Image invalide.
    HTTPException 500
        Erreur interne non anticipee.
    """
    try:
        image_bytes: bytes = file.file.read()
        processor: BlipProcessor = request.app.state.blip_processor
        model: BlipForConditionalGeneration = request.app.state.blip_model
        description = _vision_service.describe_environment(image_bytes, processor, model)
        return DescriptionResponse(description=description)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Une erreur interne s'est produite lors du traitement de l'image.",
        ) from exc
