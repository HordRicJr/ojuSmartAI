import logging
import re
from typing import Any, Dict, List, Tuple, cast

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN
from PIL import Image
from timm.data import create_transform, resolve_model_data_config
from transformers import (
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
    BlipProcessor,
    DetrForObjectDetection,
    DetrImageProcessor,
    pipeline,
)

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
    - Salesforce/blip-vqa-base : VQA interactif et detection de monnaie.
    - facebook/detr-resnet-50 : detection d'objets multi-classes pour l'analyse de scene.

    Retourne
    --------
    Dict[str, Any]
        Cles : "device", "signature_model", "signature_transform",
        "face_detector", "emotion_pipeline", "blip_processor", "blip_model",
        "blip_vqa_processor", "blip_vqa_model", "detr_processor", "detr_model".
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

    logger.info("Chargement de Salesforce/blip-vqa-base (VQA + monnaie)...")
    blip_vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    blip_vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    blip_vqa_model.eval()
    cast(torch.nn.Module, blip_vqa_model).to(device)
    logger.info("BLIP-VQA charge avec succes.")

    logger.info("Chargement de facebook/detr-resnet-50 (detection d'objets / scene)...")
    detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    detr_model.eval()
    cast(torch.nn.Module, detr_model).to(device)
    logger.info("DETR charge avec succes.")

    return {
        "device": device,
        "signature_model": signature_model,
        "signature_transform": signature_transform,
        "face_detector": face_detector,
        "emotion_pipeline": emotion_pipe,
        "blip_processor": blip_processor,
        "blip_model": blip_model,
        "blip_vqa_processor": blip_vqa_processor,
        "blip_vqa_model": blip_vqa_model,
        "detr_processor": detr_processor,
        "detr_model": detr_model,
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

    # ------------------------------------------------------------------
    # Currency detection
    # ------------------------------------------------------------------

    # Mapping of currency keywords found in BLIP-VQA answers to ISO 4217 codes.
    _CURRENCY_MAP: Dict[str, str] = {
        "franc cfa": "XOF",
        "cfa": "XOF",
        "xof": "XOF",
        "euro": "EUR",
        "eur": "EUR",
        "dollar": "USD",
        "usd": "USD",
        "pound": "GBP",
        "gbp": "GBP",
        "naira": "NGN",
        "ngn": "NGN",
        "dirham": "AED",
        "aed": "AED",
    }

    def _decode_vqa(
        self,
        pil_image: Image.Image,
        question: str,
        processor: BlipProcessor,
        model: BlipForQuestionAnswering,
    ) -> Tuple[str, float]:
        """Soumet une question a BLIP-VQA et retourne (reponse, confiance).

        La confiance est estimee via softmax sur les logits du premier token genere.

        Parametres
        ----------
        pil_image : PIL.Image.Image
            Image RGB deja decodee.
        question : str
            Question en langage naturel.
        processor : BlipProcessor
            Processeur BLIP-VQA charge via app.state.
        model : BlipForQuestionAnswering
            Modele BLIP-VQA charge via app.state.

        Retourne
        --------
        Tuple[str, float]
            (reponse_texte, score_confiance).
        """
        device: torch.device = next(model.parameters()).device
        _proc: Any = processor
        inputs: Dict[str, Any] = _proc(images=pil_image, text=question, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            raw_outputs: Any = model.generate(
                **inputs,
                max_new_tokens=20,
                output_scores=True,
                return_dict_in_generate=True,
            )

        answer: str = _proc.decode(raw_outputs.sequences[0], skip_special_tokens=True)

        # Confidence: softmax over the first generated token scores.
        confidence: float = 0.0
        scores_list: Any = getattr(raw_outputs, "scores", None)
        if scores_list:
            first_token_scores: torch.Tensor = scores_list[0][0]
            probs = F.softmax(first_token_scores, dim=-1)
            confidence = round(float(probs.max().item()), 4)

        return answer.strip(), confidence

    def detect_currency(
        self,
        image_bytes: bytes,
        processor: BlipProcessor,
        model: BlipForQuestionAnswering,
    ) -> Tuple[str, str, float]:
        """Detecte la devise et la denomination d'un billet ou d'une piece.

        Utilise BLIP-VQA avec deux questions : denomination puis devise.
        Normalise la devise vers un code ISO 4217 quand possible.

        Parametres
        ----------
        image_bytes : bytes
            Octets bruts de l'image contenant le billet ou la piece.
        processor : BlipProcessor
            Processeur BLIP-VQA charge via app.state.
        model : BlipForQuestionAnswering
            Modele BLIP-VQA charge via app.state.

        Retourne
        --------
        Tuple[str, str, float]
            (code_devise, denomination, confiance).

        Leve
        ----
        ValueError
            Si l'image est invalide ou illisible.
        """
        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(
                "Le fichier fourni n'est pas une image valide ou son format "
                "n'est pas pris en charge."
            )
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        denomination_answer, confidence = self._decode_vqa(
            pil_image,
            "What is the denomination or face value of this banknote or coin?",
            processor,
            model,
        )

        currency_answer, _ = self._decode_vqa(
            pil_image,
            "What currency is shown in this image? Answer with the currency name.",
            processor,
            model,
        )

        # Extract numeric denomination.
        numbers = re.findall(r"\d[\d\s]*", denomination_answer)
        denomination = numbers[0].strip() if numbers else denomination_answer.strip()

        # Normalize currency to ISO 4217.
        currency_lower = currency_answer.lower()
        currency_code = "UNKNOWN"
        for keyword, code in self._CURRENCY_MAP.items():
            if keyword in currency_lower:
                currency_code = code
                break

        return currency_code, denomination, confidence

    # ------------------------------------------------------------------
    # Visual Question Answering
    # ------------------------------------------------------------------

    def answer_question(
        self,
        image_bytes: bytes,
        question: str,
        processor: BlipProcessor,
        model: BlipForQuestionAnswering,
    ) -> Tuple[str, float]:
        """Repond a une question libre sur une image via BLIP-VQA.

        Parametres
        ----------
        image_bytes : bytes
            Octets bruts de l'image.
        question : str
            Question posee par l'utilisateur (en anglais).
        processor : BlipProcessor
            Processeur BLIP-VQA charge via app.state.
        model : BlipForQuestionAnswering
            Modele BLIP-VQA charge via app.state.

        Retourne
        --------
        Tuple[str, float]
            (reponse, score_confiance).

        Leve
        ----
        ValueError
            Si l'image est invalide ou si la question est vide.
        """
        if not question or not question.strip():
            raise ValueError("La question ne peut pas etre vide.")

        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(
                "Le fichier fourni n'est pas une image valide ou son format "
                "n'est pas pris en charge."
            )
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        return self._decode_vqa(pil_image, question.strip(), processor, model)

    # ------------------------------------------------------------------
    # Scene analysis / Guide mode
    # ------------------------------------------------------------------

    def analyze_scene(
        self,
        image_bytes: bytes,
        detr_processor: DetrImageProcessor,
        detr_model: DetrForObjectDetection,
        detection_threshold: float = 0.7,
        max_objects: int = 10,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Detecte les objets presents dans la scene et genere un conseil de navigation.

        Pipeline :
        1. Decode l'image et soumet a DETR avec seuillage de confiance.
        2. Pour chaque detection retenue, calcule la position horizontale
           (gauche / centre / droite) et la proximite estimee (near / medium / far)
           d'apres la surface relative de la boite englobante.
        3. Genere un navigation_hint en anglais base sur les objets dominants.

        Parametres
        ----------
        image_bytes : bytes
            Octets bruts de l'image de scene.
        detr_processor : DetrImageProcessor
            Processeur DETR charge via app.state.
        detr_model : DetrForObjectDetection
            Modele DETR charge via app.state.
        detection_threshold : float
            Seuil de confiance minimal pour retenir une detection.
        max_objects : int
            Nombre maximal d'objets retournes, tries par confiance decroissante.

        Retourne
        --------
        Tuple[List[Dict[str, Any]], str]
            (liste_objets, navigation_hint).

        Leve
        ----
        ValueError
            Si l'image est invalide ou illisible.
        """
        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(
                "Le fichier fourni n'est pas une image valide ou son format "
                "n'est pas pris en charge."
            )
        img_h, img_w = image.shape[:2]
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        device: torch.device = next(detr_model.parameters()).device
        _detr_proc: Any = detr_processor
        inputs: Dict[str, Any] = _detr_proc(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = detr_model(**inputs)

        # Post-process: map logits + boxes to predictions with threshold.
        # Pass as list of tuples to satisfy the processor's type signature.
        target_sizes_list: list[tuple[int, int]] = [(img_h, img_w)]
        results: Dict[str, Any] = _detr_proc.post_process_object_detection(
            outputs,
            threshold=detection_threshold,
            target_sizes=target_sizes_list,
        )[0]

        scores: List[float] = results["scores"].cpu().tolist()
        labels: List[int] = results["labels"].cpu().tolist()
        boxes: List[List[float]] = results["boxes"].cpu().tolist()

        id2label: Any = detr_model.config.id2label or {}

        detected: List[Dict[str, Any]] = []
        img_area = img_w * img_h

        for score, label_id, box in zip(scores, labels, boxes):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2.0

            # Horizontal position
            if cx < img_w / 3:
                position = "left"
            elif cx < 2 * img_w / 3:
                position = "center"
            else:
                position = "right"

            # Proximity estimation from bbox area relative to image area.
            bbox_area = max(0.0, (x2 - x1) * (y2 - y1))
            area_ratio = bbox_area / img_area
            if area_ratio >= 0.15:
                proximity = "near"
            elif area_ratio >= 0.04:
                proximity = "medium"
            else:
                proximity = "far"

            detected.append(
                {
                    "label": id2label.get(label_id, str(label_id)),
                    "position": position,
                    "proximity": proximity,
                    "confidence": round(score, 4),
                }
            )

        # Sort by confidence descending and cap.
        detected.sort(key=lambda d: d["confidence"], reverse=True)
        detected = detected[:max_objects]

        # Build navigation hint from the nearest/highest-confidence objects.
        navigation_hint = self._build_navigation_hint(detected)

        return detected, navigation_hint

    @staticmethod
    def _build_navigation_hint(objects: List[Dict[str, Any]]) -> str:
        """Genere un conseil de navigation en anglais a partir des objets detectes.

        Priorise les objets proches, puis enumere les autres selon leur position.
        """
        if not objects:
            return "The path ahead appears clear."

        near_objects = [o for o in objects if o["proximity"] == "near"]
        if near_objects:
            labels = [o["label"] for o in near_objects[:3]]
            positions = [o["position"] for o in near_objects[:3]]
            if len(labels) == 1:
                return (
                    f"Caution: {labels[0]} detected nearby on your {positions[0]}."
                )
            parts = ", ".join(
                f"{lbl} on your {pos}" for lbl, pos in zip(labels, positions)
            )
            return f"Caution: nearby objects detected — {parts}."

        # No near objects — report the dominant center obstacle if any.
        center = [o for o in objects if o["position"] == "center"]
        if center:
            top = center[0]
            return (
                f"{top['label'].capitalize()} detected straight ahead ({top['proximity']}). "
                "Proceed with care."
            )

        return "Scene is clear ahead. Objects detected on the sides."

    # ------------------------------------------------------------------
    # Signature comparison (anti-fraud)
    # ------------------------------------------------------------------

    def compare_signatures(
        self,
        image_bytes_1: bytes,
        image_bytes_2: bytes,
        model: torch.nn.Module,
        transform: Any,
        threshold: float = 0.92,
    ) -> Tuple[float, bool, str]:
        """Compare deux signatures par similarite cosinus sur leurs embeddings ResNet50.

        Parametres
        ----------
        image_bytes_1 : bytes
            Octets bruts de la premiere signature (reference).
        image_bytes_2 : bytes
            Octets bruts de la deuxieme signature (a verifier).
        model : torch.nn.Module
            ResNet50 timm sans couche FC, charge via app.state.
        transform : Any
            Transformation timm (resize + normalisation ImageNet).
        threshold : float
            Seuil de similarite cosinus au-dessus duquel les signatures sont jugees
            authentiques. Configurable via settings.signature_similarity_threshold.

        Retourne
        --------
        Tuple[float, bool, str]
            (similarite, is_authentic, verdict).

        Leve
        ----
        ValueError
            Si l'une des images est invalide ou illisible.
        """
        pil_1 = self.preprocess_signature(image_bytes_1)
        pil_2 = self.preprocess_signature(image_bytes_2)

        vec_1 = np.array(self.get_signature_embedding(pil_1, model, transform))
        vec_2 = np.array(self.get_signature_embedding(pil_2, model, transform))

        norm_1 = np.linalg.norm(vec_1)
        norm_2 = np.linalg.norm(vec_2)

        if norm_1 == 0.0 or norm_2 == 0.0:
            return 0.0, False, "SUSPICIOUS"

        similarity = float(np.dot(vec_1, vec_2) / (norm_1 * norm_2))
        # Clamp to [0, 1] to handle floating-point edge cases.
        similarity = round(max(0.0, min(1.0, similarity)), 4)

        is_authentic = similarity >= threshold
        verdict = "AUTHENTIC" if is_authentic else "SUSPICIOUS"

        return similarity, is_authentic, verdict