from fastapi import APIRouter, Form, HTTPException, Request, UploadFile, status
from transformers import BlipForConditionalGeneration, BlipProcessor

from app.core.config import settings
from app.models.schemas import (
    CurrencyResponse,
    DescriptionResponse,
    EmbeddingResponse,
    EmotionResponse,
    SceneResponse,
    SignatureCompareResponse,
    VQAResponse,
)
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


@router.post(
    "/currency/detect",
    response_model=CurrencyResponse,
    summary="Detecter la devise et la denomination d'un billet ou d'une piece",
    description=(
        "Recoit une image en multipart/form-data et identifie la devise "
        "(Franc CFA, USD, EUR, etc.) ainsi que la valeur faciale via BLIP-VQA. "
        "Retourne le code ISO 4217 de la devise et la denomination sous forme de chaine."
    ),
)
def detect_currency(request: Request, file: UploadFile) -> CurrencyResponse:
    """Point d'entree pour la detection de monnaie.

    Synchrone afin que FastAPI l'execute dans un threadpool (inference BLIP-VQA bloquante).

    Parametres
    ----------
    request : Request
        Acces a app.state (blip_vqa_processor, blip_vqa_model).
    file : UploadFile
        Image transmise en multipart/form-data.

    Retourne
    --------
    CurrencyResponse
        Code devise ISO 4217, denomination et score de confiance.

    Leve
    ----
    HTTPException 400
        Image invalide ou format non supporte.
    HTTPException 500
        Erreur interne non anticipee.
    """
    try:
        image_bytes: bytes = file.file.read()
        currency, denomination, confidence = _vision_service.detect_currency(
            image_bytes,
            request.app.state.blip_vqa_processor,
            request.app.state.blip_vqa_model,
        )
        return CurrencyResponse(
            currency=currency,
            denomination=denomination,
            confidence=confidence,
        )
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
    "/vqa/answer",
    response_model=VQAResponse,
    summary="Repondre a une question sur une image (Mode Guide Chien)",
    description=(
        "Recoit une image et une question en multipart/form-data. "
        "Genere une reponse textuelle via BLIP-VQA (Salesforce/blip-vqa-base). "
        "Usage : 'What is in front of me?', 'Where am I?', 'What color is the door?'."
    ),
)
def answer_vqa(
    request: Request,
    file: UploadFile,
    question: str = Form(..., description="Question en anglais posee sur l'image."),
) -> VQAResponse:
    """Point d'entree pour le VQA interactif (Mode Guide Chien).

    Synchrone afin que FastAPI l'execute dans un threadpool (inference BLIP-VQA bloquante).

    Parametres
    ----------
    request : Request
        Acces a app.state (blip_vqa_processor, blip_vqa_model).
    file : UploadFile
        Image transmise en multipart/form-data.
    question : str
        Question libre en langage naturel (champ Form).

    Retourne
    --------
    VQAResponse
        Reponse textuelle et score de confiance.

    Leve
    ----
    HTTPException 400
        Image invalide, format non supporte ou question vide.
    HTTPException 500
        Erreur interne non anticipee.
    """
    try:
        image_bytes: bytes = file.file.read()
        answer, confidence = _vision_service.answer_question(
            image_bytes,
            question,
            request.app.state.blip_vqa_processor,
            request.app.state.blip_vqa_model,
        )
        return VQAResponse(answer=answer, confidence=confidence)
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
    "/guide/scene",
    response_model=SceneResponse,
    summary="Analyser la scene et generer un conseil de navigation",
    description=(
        "Recoit une image en multipart/form-data et detecte les objets presents "
        "via DETR (facebook/detr-resnet-50). "
        "Retourne la liste des objets avec leur position et proximite, "
        "ainsi qu'un conseil de navigation en anglais."
    ),
)
def analyze_scene(request: Request, file: UploadFile) -> SceneResponse:
    """Point d'entree pour l'analyse de scene / Mode Guide Chien.

    Synchrone afin que FastAPI l'execute dans un threadpool (inference DETR bloquante).

    Parametres
    ----------
    request : Request
        Acces a app.state (detr_processor, detr_model).
    file : UploadFile
        Image transmise en multipart/form-data.

    Retourne
    --------
    SceneResponse
        Liste d'objets detectes, conseil de navigation et nombre d'objets.

    Leve
    ----
    HTTPException 400
        Image invalide ou format non supporte.
    HTTPException 500
        Erreur interne non anticipee.
    """
    try:
        image_bytes: bytes = file.file.read()
        objects, navigation_hint = _vision_service.analyze_scene(
            image_bytes,
            request.app.state.detr_processor,
            request.app.state.detr_model,
            detection_threshold=settings.scene_detection_threshold,
            max_objects=settings.scene_max_objects,
        )
        return SceneResponse(
            objects=objects,
            navigation_hint=navigation_hint,
            object_count=len(objects),
        )
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
    "/signatures/compare",
    response_model=SignatureCompareResponse,
    summary="Comparer deux signatures pour detecter une fraude",
    description=(
        "Recoit deux images de signatures en multipart/form-data (file1, file2). "
        "Calcule la similarite cosinus entre leurs embeddings ResNet50. "
        "Retourne le score de similarite, un booleen is_authentic et un verdict."
    ),
)
def compare_signatures(
    request: Request,
    file1: UploadFile,
    file2: UploadFile,
) -> SignatureCompareResponse:
    """Point d'entree pour la comparaison anti-fraude de signatures.

    Synchrone afin que FastAPI l'execute dans un threadpool (two ResNet50 forward passes).

    Parametres
    ----------
    request : Request
        Acces a app.state (signature_model, signature_transform).
    file1 : UploadFile
        Image de la signature de reference (multipart/form-data).
    file2 : UploadFile
        Image de la signature a verifier (multipart/form-data).

    Retourne
    --------
    SignatureCompareResponse
        Similarite cosinus, verdict et booleen is_authentic.

    Leve
    ----
    HTTPException 400
        Image invalide ou format non supporte.
    HTTPException 500
        Erreur interne non anticipee.
    """
    try:
        bytes_1: bytes = file1.file.read()
        bytes_2: bytes = file2.file.read()
        similarity, is_authentic, verdict = _vision_service.compare_signatures(
            bytes_1,
            bytes_2,
            request.app.state.signature_model,
            request.app.state.signature_transform,
            threshold=settings.signature_similarity_threshold,
        )
        return SignatureCompareResponse(
            similarity=similarity,
            is_authentic=is_authentic,
            verdict=verdict,
        )
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
