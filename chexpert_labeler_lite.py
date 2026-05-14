"""
chexpert_labeler_lite.py — Implementación ligera del CheXpert Labeler.

Replica la lógica oficial de Stanford (Irvin et al., 2019) en Python puro:
    • 14 etiquetas estándar de CheXpert
    • Detección de mención mediante frases/patrones por etiqueta
    • Detección de negación e incertidumbre vía regex
    • Valores de salida:
        1.0  → Positivo (mencionado, no negado, no incierto)
        0.0  → Negativo (mencionado y negado)
       -1.0  → Incierto (mencionado con hedge/incertidumbre)
        NaN  → No mencionado

Referencias:
    Irvin, J. et al. "CheXpert: A Large Chest Radiograph Dataset with Uncertainty
    Labels and Expert Comparison." AAAI 2019.
"""

import re
import math
from typing import Dict, Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Patrones de negación (adaptados del paper CheXpert)
# ──────────────────────────────────────────────────────────────────────────────
_NEG_PATTERN = re.compile(
    r"\b(no|not|without|free of|absence of|absent|negative for|"
    r"no evidence of|no sign of|no signs of|ruled out|unremarkable|"
    r"clear of|denies|denied|excluding|exclude)\b",
    re.IGNORECASE
)

# Patrones de incertidumbre (CheXpert los llama "uncertainty" o "hedging")
_UNC_PATTERN = re.compile(
    r"\b(possible|possibly|probable|probably|may|might|could represent|"
    r"cannot exclude|questionable|suspected|suspicion of|suspect|"
    r"consistent with|cannot rule out|likely|unlikely|appears|"
    r"suggestion of|suggestive of|concern for|worrisome)\b",
    re.IGNORECASE
)

# Ventana de contexto (palabras) hacia atrás desde la mención
_CONTEXT_WINDOW = 8


# ──────────────────────────────────────────────────────────────────────────────
# Frases de mención por etiqueta (mention phrases del paper CheXpert)
# ──────────────────────────────────────────────────────────────────────────────
_MENTION_PATTERNS: Dict[str, re.Pattern] = {
    "No Finding": re.compile(
        r"\b(no acute|normal|unremarkable|no abnormality|no acute cardiopulmonary|"
        r"no acute findings|no acute disease)\b",
        re.IGNORECASE
    ),
    "Enlarged Cardiomediastinum": re.compile(
        r"\b(enlarged cardiomediastin|widened mediastin|mediastinal widening|"
        r"prominent mediastin)\b",
        re.IGNORECASE
    ),
    "Cardiomegaly": re.compile(
        r"\b(cardiomegaly|enlarged heart|cardiac enlargement|"
        r"increased cardiac|enlarged cardiac silhouette|"
        r"cardiomegal)\b",
        re.IGNORECASE
    ),
    "Lung Lesion": re.compile(
        r"\b(lung lesion|pulmonary lesion|lung mass|pulmonary mass|"
        r"lung nodule|pulmonary nodule|nodular opacity|"
        r"focal lung|lung neoplasm)\b",
        re.IGNORECASE
    ),
    "Lung Opacity": re.compile(
        r"\b(lung opacity|pulmonary opacity|opacit|airspace opacity|"
        r"parenchymal opacity|ground.glass|haziness|hazy)\b",
        re.IGNORECASE
    ),
    "Edema": re.compile(
        r"\b(edema|pulmonary edema|pulmonary vascular congestion|"
        r"vascular congestion|interstitial edema|alveolar edema|"
        r"pulmonary venous hypertension|fluid overload)\b",
        re.IGNORECASE
    ),
    "Consolidation": re.compile(
        r"\b(consolidation|consolidat|airspace disease|alveolar filling|"
        r"lobar consolidation|segmental consolidation)\b",
        re.IGNORECASE
    ),
    "Pneumonia": re.compile(
        r"\b(pneumonia|pneumonic|infectious process|"
        r"aspiration pneumonia|community.acquired pneumonia)\b",
        re.IGNORECASE
    ),
    "Atelectasis": re.compile(
        r"\b(atelectasis|atelectat|atelectatic|collapse|discoid|"
        r"subsegmental atelectasis|plate.like atelectasis|"
        r"linear atelectasis|basilar atelectasis)\b",
        re.IGNORECASE
    ),
    "Pneumothorax": re.compile(
        r"\b(pneumothorax|pneumothoraces|air in the pleural|"
        r"pleural air|tension pneumothorax)\b",
        re.IGNORECASE
    ),
    "Pleural Effusion": re.compile(
        r"\b(pleural effusion|pleural fluid|fluid in the pleural|"
        r"pleural collection|hydrothorax|effusion)\b",
        re.IGNORECASE
    ),
    "Pleural Other": re.compile(
        r"\b(pleural thickening|pleural calcification|pleural plaque|"
        r"pleural scarring|fibrothorax|pleural disease|"
        r"pleural abnormality)\b",
        re.IGNORECASE
    ),
    "Fracture": re.compile(
        r"\b(fracture|fractured|rib fracture|clavicle fracture|"
        r"vertebral fracture|compression fracture|"
        r"acute fracture|chronic fracture)\b",
        re.IGNORECASE
    ),
    "Support Devices": re.compile(
        r"\b(endotracheal tube|et tube|ett|nasogastric tube|ng tube|"
        r"central line|central venous catheter|pacemaker|icd|"
        r"defibrillator|chest tube|picc|port.a.cath|tracheostomy|"
        r"enteric tube|feeding tube|support device|monitoring lead)\b",
        re.IGNORECASE
    ),
}


def _context_before(text: str, match_start: int, n_words: int = _CONTEXT_WINDOW) -> str:
    """Extrae las n palabras antes del inicio del match."""
    prefix = text[:match_start]
    tokens = prefix.split()
    return " ".join(tokens[-n_words:])


def _label_single(text: str, mention_re: re.Pattern) -> float:
    """
    Etiqueta una sola condición en el texto.

    Returns:
        1.0  → positivo
        0.0  → negativo
       -1.0  → incierto
        NaN  → no mencionado
    """
    match = mention_re.search(text)
    if match is None:
        return float("nan")

    context = _context_before(text, match.start())

    if _NEG_PATTERN.search(context):
        return 0.0   # Negativo

    if _UNC_PATTERN.search(context):
        return -1.0  # Incierto

    return 1.0       # Positivo


class CheXpertLabelerLite:
    """
    Labeler CheXpert ligero basado en reglas (sin Java, sin servidor).

    Implementa la lógica de menciones + negación + incertidumbre
    del paper original para las 14 etiquetas estándar.

    Valores de salida:
        1.0  → Positivo
        0.0  → Negativo
       -1.0  → Incierto
        NaN  → No mencionado

    Example:
        >>> labeler = CheXpertLabelerLite()
        >>> labels = labeler.label("No pleural effusion or pneumothorax.")
        >>> print(labels["Pleural Effusion"])   # 0.0
        >>> print(labels["Pneumothorax"])       # 0.0
    """

    LABELS = list(_MENTION_PATTERNS.keys())

    def label(self, report: str) -> Dict[str, float]:
        """
        Etiqueta un reporte con las 14 categorías CheXpert.

        Args:
            report: Texto del reporte médico (Findings + Impression).

        Returns:
            Diccionario {etiqueta: valor} donde valor ∈ {1.0, 0.0, -1.0, NaN}.
        """
        result: Dict[str, float] = {}
        for label_name, mention_re in _MENTION_PATTERNS.items():
            result[label_name] = _label_single(report, mention_re)
        return result

    def label_batch(self, reports: list) -> list:
        """
        Etiqueta una lista de reportes.

        Args:
            reports: Lista de strings de reportes.

        Returns:
            Lista de diccionarios {etiqueta: valor}.
        """
        return [self.label(r) for r in reports]

    def to_vector(self, labels_dict: Dict[str, float]) -> list:
        """
        Convierte el diccionario de etiquetas a un vector numérico ordenado.

        El orden sigue self.LABELS (14 elementos).
        NaN se mantiene como float('nan').

        Args:
            labels_dict: Resultado de self.label().

        Returns:
            Lista de 14 floats.
        """
        return [labels_dict[lbl] for lbl in self.LABELS]
