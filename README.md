---
title: OjuSmartAI
emoji: 👁
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# OjuSmart AI Engine

Microservice d inference IA stateless pour la plateforme **OjuSmart**.
Construit avec FastAPI et PyTorch, expose trois endpoints de vision par ordinateur consommes par le backend Spring Boot.

## Endpoints

| Methode | Route | Description |
|---------|-------|-------------|
| POST | /ai/v1/signatures/analyze | Embedding 2048 dim via ResNet50 (timm) |
| POST | /ai/v1/emotions/detect | Emotion dominante via MTCNN + dima806 |
| POST | /ai/v1/environment/describe | Description texte via BLIP |
| GET | /health | Etat du service + dispositif de calcul |
| GET | /docs | Swagger UI |

## Modeles utilises

| Modele | Source | Usage |
|--------|--------|-------|
| resnet50.a1_in1k | timm | Embeddings signatures |
| dima806/facial_emotions_image_detection | Hugging Face | Classification emotions (7 classes) |
| Salesforce/blip-image-captioning-base | Hugging Face | Description visuelle |
| MTCNN | facenet-pytorch | Detection visages |

## Stack technique

- **FastAPI** - Framework API asynchrone
- **PyTorch** - Inference deep learning
- **timm** - Vision models (ResNet50)
- **transformers** - Pipelines Hugging Face
- **OpenCV** - Pretraitement images
- **facenet-pytorch** - Detection MTCNN

## Exemple

curl -X POST https://hordricjr-ojusmartai.hf.space/ai/v1/emotions/detect -F "file=@photo.jpg"

Reponse : { "emotion": "happy", "confidence": 0.9823 }