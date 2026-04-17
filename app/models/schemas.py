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
