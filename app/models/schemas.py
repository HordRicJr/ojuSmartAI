from typing import List

from pydantic import BaseModel, Field


class EmbeddingResponse(BaseModel):
    """Réponse contenant le vecteur d'embedding d'une signature."""

    embedding: List[float] = Field(
        ...,
        description="Vecteur de caractéristiques extrait de l'image de signature.",
    )
    dimension: int = Field(
        ...,
        description="Nombre de dimensions du vecteur d'embedding.",
    )


class EmotionResponse(BaseModel):
    """Réponse contenant l'émotion détectée sur une image."""

    emotion: str = Field(
        ...,
        description="Libellé de l'émotion détectée (ex: neutre, sourire).",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score de confiance de la détection, compris entre 0 et 1.",
    )


class DescriptionResponse(BaseModel):
    """Reponse contenant la description textuelle generee a partir d'une image."""

    description: str = Field(
        ...,
        description="Phrase descriptive generee par le modele BLIP a partir de l'image.",
    )


class CurrencyResponse(BaseModel):
    """Reponse contenant la devise et la denomination detectees sur un billet ou une piece."""

    currency: str = Field(
        ...,
        description="Code ISO 4217 de la devise detectee (ex: 'XOF', 'EUR', 'USD').",
    )
    denomination: str = Field(
        ...,
        description="Valeur faciale du billet ou de la piece (ex: '1000', '5000', '500').",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score de confiance de la detection, compris entre 0 et 1.",
    )


class VQAResponse(BaseModel):
    """Reponse contenant la reponse generee par le modele VQA a une question sur une image."""

    answer: str = Field(
        ...,
        description="Reponse textuelle generee par BLIP-VQA a la question posee.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score de confiance estime a partir des logits du premier token genere.",
    )


class DetectedObject(BaseModel):
    """Objet detecte dans la scene par DETR."""

    label: str = Field(
        ...,
        description="Categorie de l'objet detecte (ex: 'person', 'car', 'chair').",
    )
    position: str = Field(
        ...,
        description="Position horizontale de l'objet dans l'image : 'left', 'center' ou 'right'.",
    )
    proximity: str = Field(
        ...,
        description="Proximite estimee de l'objet : 'near', 'medium' ou 'far'.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score de confiance de la detection DETR.",
    )


class SceneResponse(BaseModel):
    """Reponse contenant les objets detectes et un conseil de navigation."""

    objects: List[DetectedObject] = Field(
        ...,
        description="Liste des objets detectes dans la scene, triee par confiance decroissante.",
    )
    navigation_hint: str = Field(
        ...,
        description="Conseil de navigation en anglais genere a partir des objets detectes.",
    )
    object_count: int = Field(
        ...,
        ge=0,
        description="Nombre total d'objets detectes dans la scene.",
    )


class SignatureCompareResponse(BaseModel):
    """Reponse contenant le resultat de la comparaison anti-fraude de deux signatures."""

    similarity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score de similarite cosinus entre les deux embeddings (0 = different, 1 = identique).",
    )
    is_authentic: bool = Field(
        ...,
        description="True si la similarite depasse le seuil de confiance configure.",
    )
    verdict: str = Field(
        ...,
        description="Verdict lisible : 'AUTHENTIC' ou 'SUSPICIOUS'.",
    )
