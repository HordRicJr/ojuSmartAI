import logging
from typing import Any, Dict, List, cast

import cv2
import numpy as np
import timm
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from timm.data import create_transform, resolve_model_data_config
from transformers import BlipForConditionalGeneration, BlipProcessor, pipeline

logger = logging.getLogger(__name__)

_TARGET_SIZE = 224


def _get_device() -> torch.device:
    """Detecte dynamiquement le meilleur dispositif de calcul disponible.

    Ordre de priorite : CUDA (GPU NVIDIA) > MPS (Apple Silicon) > CPU.

    Retourne
    --------
    torch.device
        Le dispositif de calcul le plus performant disponible sur la machine.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_models() -> Dict[str, Any]:
    """Charge tous les modeles d'IA en memoire et les retourne dans un dictionnaire.

    Destinee a etre appelee une seule fois au demarrage via le lifespan FastAPI.
    Les modeles sont stockes dans app.state pour etre reinjectes sans rechargement.

    Modeles charges :
    - timm/resnet50.a1_in1k : extraction d'embeddings de signatures (2048 dim).
    - dima806/facial_emotions_image_detection : classification d'emotions (7 classes).
    - Salesforce/blip-image-captioning-base : description visuelle en langage naturel.

    Retourne
    --------
    Dict[str, Any]
        Cles : "device", "signature_model", "signature_transform",
        "face_detector", "emotion_pipeline", "blip_processor", "blip_model".
    """
    device = _get_device()
    logger.info("Dispositif de calcul detecte : %s", device)

    logger.info("Chargement de resnet50.a1_in1k (signatures)...")
    signature_model = timm.create_model(
        "resnet50.a1_in1k",
        pretrained=True,
        num_classes=0,
    )
    signature_model.eval()
    signature_model.to(device)
    data_config = resolve_model_data_config(signature_model)
    signature_transform = create_transform(**data_config, is_training=False)
    logger.info("ResNet50 timm charge avec succes.")

    logger.info("Chargement de MTCNN (detecteur de visages)...")
    face_detector = MTCNN(keep_all=True, device=device)
    logger.info("MTCNN charge avec succes.")

    logger.info("Chargement de dima806/facial_emotions_image_detection (emotions)...")
    emotion_pipe = pipeline(
        "image-classification",
        model="dima806/facial_emotions_image_detection",
        device=device,
    )
    logger.info("Pipeline emotions charge avec succes.")

    logger.info("Chargement de Salesforce/blip-image-captioning-base (description)...")
    blip_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    blip_model.eval()
    cast(torch.nn.Module, blip_model).to(device)
    logger.info("BLIP charge avec succes.")

    return {
        "device": device,
        "signature_model": signature_model,
        "signature_transform": signature_transform,
        "face_detector": face_detector,
        "emotion_pipeline": emotion_pipe,
        "blip_processor": blip_processor,
        "blip_model": blip_model,
    }


class VisionService:
    """Service de traitement d'images et d'inference visuelle.

    Stateless : aucune connexion a une base de donnees. Les modeles sont
    injectes en parametre depuis app.state pour ne charger qu'une seule fois.
    """

    def preprocess_signature(self, image_bytes: bytes) -> Image.Image:
        """Decode et prepare une image de signature pour l'inference ResNet50.

        Pipeline :
        1. Decodage via cv2.imdecode.
        2. Conversion en niveaux de gris.
        3. Seuillage Otsu inverse pour isoler les traits.
        4. Recadrage sur la boite englobante du plus grand contour.
        5. Redimensionnement 224x224.
        6. Conversion en PIL Image RGB pour timm.

        Parametres
        ----------
        image_bytes : bytes
            Octets bruts de l'image de signature.

        Retourne
        --------
        PIL.Image.Image
            Image RGB 224x224 prete pour la transformation timm.

        Leve
        ----
        ValueError
            Si le decodage echoue ou si le format n'est pas supporte.
        """
        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(
                "Le fichier fourni n'est pas une image valide ou son format "
                "n'est pas pris en charge."
            )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            all_contours = np.vstack(contours)
            x, y, w, h = cv2.boundingRect(all_contours)
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(gray.shape[1] - x, w + 2 * padding)
            h = min(gray.shape[0] - y, h + 2 * padding)
            cropped = gray[y : y + h, x : x + w]
        else:
            cropped = gray

        src_h, src_w = cropped.shape[:2]
        interp = (
            cv2.INTER_AREA
            if (src_h > _TARGET_SIZE or src_w > _TARGET_SIZE)
            else cv2.INTER_LINEAR
        )
        resized = cv2.resize(cropped, (_TARGET_SIZE, _TARGET_SIZE), interpolation=interp)
        rgb_array = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb_array)

    def get_signature_embedding(
        self,
        pil_image: Image.Image,
        model: torch.nn.Module,
        transform: Any,
    ) -> List[float]:
        """Extrait un vecteur d'embedding via ResNet50 timm.

        Applique la transformation timm, effectue le passage avant sans gradient
        et retourne le vecteur de 2048 caracteristiques.

        Parametres
        ----------
        pil_image : PIL.Image.Image
            Image preprocessee par preprocess_signature.
        model : torch.nn.Module
            ResNet50 timm sans couche FC, charge via app.state.
        transform : Any
            Transformation timm (resize + normalisation ImageNet).

        Retourne
        --------
        List[float]
            Vecteur d'embedding de dimension 2048.
        """
        tensor: torch.Tensor = transform(pil_image).unsqueeze(0)
        device: torch.device = next(model.parameters()).device
        tensor = tensor.to(device)

        with torch.no_grad():
            output: torch.Tensor = model(tensor)

        return output.squeeze().flatten().tolist()

    def analyze_emotion(
        self,
        image_bytes: bytes,
        emotion_pipeline: Any,
        face_detector: Any,
    ) -> tuple[str, float]:
        """Detecte l'emotion dominante sur le visage le plus grand de l'image.

        Pipeline :
        1. Decodage de l'image.
        2. Detection des visages via MTCNN.
        3. Selection du visage le plus grand (surface bbox maximale).
        4. Recadrage strict sur ce visage.
        5. Classification d'emotion via dima806/facial_emotions_image_detection.

        Parametres
        ----------
        image_bytes : bytes
            Octets bruts de l'image.
        emotion_pipeline : Any
            Pipeline HF "image-classification" charge via app.state.
        face_detector : Any
            Modele MTCNN charge via app.state.

        Retourne
        --------
        tuple[str, float]
            (libelle_emotion, score_confiance) avec score entre 0.0 et 1.0.

        Leve
        ----
        ValueError
            Si image invalide, aucun visage detecte ou aucun resultat retourne.
        """
        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(
                "Le fichier fourni n'est pas une image valide ou son format "
                "n'est pas pris en charge."
            )

        rgb_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_array)

        boxes, _ = face_detector.detect(pil_image)

        if boxes is None or len(boxes) == 0:
            raise ValueError("Aucun visage detecte dans l'image fournie.")

        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        largest_idx = int(np.argmax(areas))
        x1, y1, x2, y2 = [int(c) for c in boxes[largest_idx]]

        img_h, img_w = rgb_array.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)

        face_crop = rgb_array[y1:y2, x1:x2]
        face_pil = Image.fromarray(face_crop)

        results: List[Dict[str, Any]] = emotion_pipeline(face_pil)

        if not results:
            raise ValueError(
                "Aucune emotion n'a pu etre detectee dans l'image fournie."
            )

        top: Dict[str, Any] = results[0]
        return top["label"], round(float(top["score"]), 4)

    def describe_environment(
        self,
        image_bytes: bytes,
        processor: BlipProcessor,
        model: BlipForConditionalGeneration,
    ) -> str:
        """Genere une description textuelle de l'image via BLIP.

        Convertit les octets en PIL Image RGB, prepare les tenseurs via
        BlipProcessor, genere les tokens et decodifie en texte.

        Parametres
        ----------
        image_bytes : bytes
            Octets bruts de l'image a decrire.
        processor : BlipProcessor
            Processeur BLIP charge via app.state.
        model : BlipForConditionalGeneration
            Modele BLIP charge via app.state.

        Retourne
        --------
        str
            Phrase descriptive generee par BLIP.

        Leve
        ----
        ValueError
            Si le decodage de l'image echoue.
        """
        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(
                "Le fichier fourni n'est pas une image valide ou son format "
                "n'est pas pris en charge."
            )

        rgb_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_array)

        device: torch.device = next(model.parameters()).device
        _proc: Any = processor
        encoding: Any = _proc(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output_ids = model.generate(**inputs, max_new_tokens=30, min_length=5)
            else:
                output_ids = model.generate(**inputs, max_new_tokens=30, min_length=5)

        description: str = processor.decode(output_ids[0], skip_special_tokens=True)
        return description