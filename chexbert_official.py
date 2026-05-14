"""
chexbert_official.py — Wrapper del modelo CheXbert oficial de Stanford.

Carga los pesos oficiales de Stanford AIMI (Smit et al., 2020) alojados en
  StanfordAIMI/RRG_scorers → chexbert.pth
y expone una interfaz limpia de labeling sobre texto de reportes radiológicos.

Arquitectura (idéntica al repositorio original stanfordmlgroup/CheXbert):
    BERT-base-uncased backbone
    + 14 linear_heads independientes (nn.ModuleList)
        · Heads 0-12  → 4 clases  {blank, positive, negative, uncertain}
        · Head 13     → 2 clases  {blank, positive}  (Support Devices)

Mapeo de clase → valor CheXpert oficial:
    Heads 0-12:  0 → NaN  |  1 → 1.0  |  2 → 0.0  |  3 → -1.0
    Head  13:    0 → NaN  |  1 → 1.0

Referencias:
    Smit, A. et al. "CheXbert: Combining Automatic Labelers and Expert Annotations
    for Accurate Radiology Report Labeling Using BERT." EMNLP 2020.
"""

import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import BertModel, BertTokenizer

# ──────────────────────────────────────────────────────────────────────────────
# 1. Constantes
# ──────────────────────────────────────────────────────────────────────────────

# Orden oficial de etiquetas (CheXpert / CheXbert paper)
CHEXBERT_LABELS: List[str] = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

# Número de clases por head según la arquitectura oficial
_NUM_CLASSES: List[int] = [4] * 13 + [2]   # heads 0-12 → 4 clases; head 13 → 2

# Ruta HuggingFace de los pesos
_HF_REPO   = "StanfordAIMI/RRG_scorers"
_HF_FILE   = "chexbert.pth"
_CACHE_DIR = os.path.expanduser("~/.cache/chexbert")

# Mapeo logit_index → valor CheXpert
#   heads 0-12 (4 clases): blank=0, pos=1, neg=2, unc=3
#   head   13  (2 clases): blank=0, pos=1
_CLASS_VALUE_4 = {0: float("nan"), 1: 1.0, 2: 0.0, 3: -1.0}
_CLASS_VALUE_2 = {0: float("nan"), 1: 1.0}


# ──────────────────────────────────────────────────────────────────────────────
# 2. Arquitectura del modelo (replica stanfordmlgroup/CheXbert bert.py)
# ──────────────────────────────────────────────────────────────────────────────

class _BertLabeler(nn.Module):
    """
    Replica exacta de la clase bert_labeler del repositorio oficial.
    https://github.com/stanfordmlgroup/CheXbert/blob/master/src/models/bert.py
    """

    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased",
            output_attentions=False,
            output_hidden_states=False,
        )
        self.dropout = nn.Dropout(p=0.1)
        self.linear_heads = nn.ModuleList(
            [nn.Linear(768, n) for n in _NUM_CLASSES]
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = self.dropout(outputs.pooler_output)   # (B, 768)
        logits = [head(cls_output) for head in self.linear_heads]
        return logits   # list of 14 tensors, each (B, num_classes_i)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Clase pública
# ──────────────────────────────────────────────────────────────────────────────

class CheXbertLabeler:
    """
    Labeler CheXbert oficial (Smit et al., 2020).

    Descarga automáticamente los pesos de StanfordAIMI/RRG_scorers si no están
    en caché y los carga sobre la arquitectura BERT-base-uncased.

    Valores de salida (mapeo oficial CheXpert):
        1.0   → Positivo   (condición mencionada, presente)
        0.0   → Negativo   (condición mencionada, ausente)
       -1.0   → Incierto   (condición mencionada con incertidumbre)
        NaN   → No mencionado / blank

    Example:
        >>> labeler = CheXbertLabeler()
        >>> labels = labeler.label("No pleural effusion or pneumothorax.")
        >>> print(labels["Pleural Effusion"])   # 0.0
        >>> print(labels["No Finding"])         # NaN  (no se menciona explícitamente)
    """

    LABELS = CHEXBERT_LABELS

    def __init__(self, device: Optional[str] = None, cache_dir: str = _CACHE_DIR):
        """
        Args:
            device   : 'cpu', 'cuda', o None (autodetecta).
            cache_dir: Directorio local para cachear chexbert.pth.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self._tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self._model     = self._load_model(cache_dir)

    # ── Carga de pesos ────────────────────────────────────────────────────────

    def _load_model(self, cache_dir: str) -> _BertLabeler:
        """Descarga (si necesario) y carga el checkpoint oficial."""
        ckpt_path = hf_hub_download(
            repo_id  = _HF_REPO,
            filename = _HF_FILE,
            cache_dir= cache_dir,
        )

        model = _BertLabeler()

        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        # El checkpoint fue guardado con DataParallel → prefijo 'module.'
        cleaned = {
            k.replace("module.", "", 1): v
            for k, v in state_dict.items()
        }

        model.load_state_dict(cleaned)
        model.eval()
        model.to(self.device)
        return model

    # ── Inferencia ────────────────────────────────────────────────────────────

    def label(self, report: str) -> Dict[str, float]:
        """
        Etiqueta un reporte con las 14 categorías CheXpert.

        Args:
            report: Texto libre del reporte radiológico.

        Returns:
            Diccionario {etiqueta: valor} donde valor ∈ {1.0, 0.0, -1.0, NaN}.
        """
        encoding = self._tokenizer(
            report,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",
        )

        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits_list = self._model(input_ids, attention_mask)

        result: Dict[str, float] = {}
        for i, (label_name, logits) in enumerate(zip(CHEXBERT_LABELS, logits_list)):
            pred_class = int(logits.argmax(dim=-1).item())
            mapping    = _CLASS_VALUE_2 if i == 13 else _CLASS_VALUE_4
            result[label_name] = mapping[pred_class]

        return result

    def label_batch(self, reports: List[str], batch_size: int = 16) -> List[Dict[str, float]]:
        """
        Etiqueta una lista de reportes en batches.

        Args:
            reports   : Lista de strings.
            batch_size: Número de reportes por lote.

        Returns:
            Lista de diccionarios {etiqueta: valor}, uno por reporte.
        """
        results = []
        for i in range(0, len(reports), batch_size):
            batch = reports[i : i + batch_size]
            encoding = self._tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length",
            )
            input_ids      = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            with torch.no_grad():
                logits_list = self._model(input_ids, attention_mask)

            batch_size_real = input_ids.shape[0]
            for sample_idx in range(batch_size_real):
                row: Dict[str, float] = {}
                for j, (label_name, logits) in enumerate(zip(CHEXBERT_LABELS, logits_list)):
                    pred_class = int(logits[sample_idx].argmax().item())
                    mapping    = _CLASS_VALUE_2 if j == 13 else _CLASS_VALUE_4
                    row[label_name] = mapping[pred_class]
                results.append(row)

        return results
