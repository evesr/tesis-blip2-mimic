#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_raddino_probe.py — Sonda diagnóstica AISLADA del codificador visual RadDINO
================================================================================
Evalúa el rendimiento diagnóstico INTRÍNSECO de RadDINO (microsoft/rad-dino) en
clasificación MULTI-ETIQUETA de las 14 patologías CheXpert de MIMIC-CXR, de forma
aislada del resto del pipeline BLIP-2.

Dos modos seleccionables con un único toggle booleano (--apply-lora):

  • apply_lora = False  →  LINEAR PROBING ESTRICTO
        RadDINO se CONGELA al 100 %. Se entrena ÚNICAMENTE un cabezal lineal
        (nn.Linear(hidden_size → 14)) conectado a la salida densa [CLS]
        (pooler_output, con fallback a last_hidden_state[:, 0]).
        → Mide la calidad "tal cual" de las representaciones de RadDINO.
        → Optimización extra: extrae las features [CLS] UNA sola vez (cache en
          disco) y entrena el cabezal sobre tensores cacheados → entrenamiento
          del cabezal casi instantáneo, batches enormes, muchas épocas.

  • apply_lora = True   →  LoRA PROBING
        Se inyectan adaptadores LoRA al backbone de RadDINO (query, value;
        misma config que el Experimento 2) ADEMÁS del cabezal lineal. El backbone
        sigue congelado salvo las matrices LoRA. Entrenamiento end-to-end.

Optimización de GPU/tiempo (exigida):
  • Precisión mixta bfloat16 (torch.autocast) — sin GradScaler (bf16 tiene rango).
  • DataLoader con prefetching multiproceso (num_workers, pin_memory,
    persistent_workers, prefetch_factor).
  • torch.inference_mode() en validación y en la extracción de features.
  • cudnn.benchmark + TF32 habilitados; tamaño de entrada fijo → batches grandes.

Reutiliza (SIN reescribir) la carga de datos del proyecto:
  • preprocessing.estandarizar_imagen  (CLAHE → RGB → resize, idéntico a Fase 2/3/Exp2)
  • data_loader.MimicCXRDataset         (solo para construir el índice O(1) de imágenes)
  • chexbert_official.CHEXBERT_LABELS   (orden canónico de las 14 clases)
  • trainer_utils_exp2_raddino          (RADDINO_MODEL_ID + hiperparámetros LoRA + rutas)
GT de patologías: mimic-cxr-2.0.0-chexpert.csv (merge por study_id).

Normalización de imagen: estadísticas NATIVAS de RadDINO (AutoImageProcessor:
mean/std ≈ 0.5307/0.2583), aplicadas sobre la PIL ya realzada con CLAHE del
proyecto. El tamaño por defecto es 224×224 (DINOv2 interpola las pos-encodings,
igual que en el Experimento 2); --image-size 518 usa la resolución nativa.

Salidas (en <output-dir>/raddino_{linear,lora}_probe/):
  • history.csv               — curva de Loss (train/val) + macro P/R/F1/AUROC por época
  • metrics_per_pathology.csv — P/R/F1/AUROC + soporte por patología (val y test)
  • metrics_summary.json      — macro/micro + config + mejor época + timing
  • loss_curve.png            — gráfico de la curva de Loss + macro-F1 (si hay matplotlib)
  • best_state.pt             — pesos del cabezal (+ LoRA si aplica)

Uso:
    # Linear probing estricto (rápido, con cache de features):
    python eval_raddino_probe.py

    # LoRA probing end-to-end:
    python eval_raddino_probe.py --apply-lora

    # Smoke test rápido:
    python eval_raddino_probe.py --num-train 512 --num-val 256 --epochs 2 --no-cache-features

    # Evaluar también en el Test Set al final:
    python eval_raddino_probe.py --eval-test

RESTRICCIÓN: NO crea entornos virtuales ni instala nada — corre sobre el Python
del sistema (mismas dependencias que el resto del proyecto).

Autor : Evelyn Silva Rozas
Fecha : Junio 2026
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# 0. Entorno (antes de importar transformers/tokenizers)
# ─────────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import gc
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms

# cv2 sin hilos para no competir con los workers del DataLoader
import cv2
cv2.setNumThreads(0)

# ── Módulos del proyecto (reutilización de la carga de datos) ────────────────
from config import config
from model_utils import cargar_procesador
from data_loader import MimicCXRDataset
from preprocessing import estandarizar_imagen, crear_imagen_placeholder
from chexbert_official import CHEXBERT_LABELS

# ── Constantes del Experimento 2 (modelo + hiperparámetros LoRA + rutas) ─────
from trainer_utils_exp2_raddino import (
    RADDINO_MODEL_ID,
    RADDINO_TARGET_MODULES,
    RADDINO_LORA_R,
    RADDINO_LORA_ALPHA,
    RADDINO_LORA_DROPOUT,
    BASE_DIR,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    IMAGES_DIR,
    OUTPUT_DIR,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("raddino_probe")

NUM_LABELS = len(CHEXBERT_LABELS)   # 14
CHEXPERT_CSV_DEFAULT = BASE_DIR / "mimic-cxr-2.0.0-chexpert.csv"
SEP = "─" * 74


# ─────────────────────────────────────────────────────────────────────────────
# 2. Reproducibilidad + flags de rendimiento
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enable_perf_flags() -> None:
    """Activa TF32 + cudnn.benchmark (entrada de tamaño fijo → máxima velocidad)."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 3. Etiquetas GT CheXpert — merge + política de incertidumbre
# ─────────────────────────────────────────────────────────────────────────────
def _norm_sid(s) -> str:
    """Normaliza study_id (quita 's' inicial + espacios). Idéntico a Fase 3/Exp2."""
    s = str(s).strip()
    return s.lstrip("s") if s.startswith("s") else s


def construir_matriz_labels(
    df_split: pd.DataFrame,
    df_chex: pd.DataFrame,
    uncertain: str = "positive",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Une el GT oficial CheXpert al split (por study_id) y devuelve:
        labels (N, 14) float32  — objetivo binario por política de incertidumbre
        masks  (N, 14) float32  — 1 = la clase cuenta en loss/métricas; 0 = ignorar
        df_ok                   — filas del split con GT disponible (alineadas)

    Codificación del GT MIMIC-CXR:
        1.0  → positivo
        0.0  → negativo
       -1.0  → incierto (mencionado, ambiguo)
        NaN  → no mencionado (se trata como negativo, convención del proyecto)

    Políticas de incertidumbre (--uncertain):
        'positive' (def.): -1 → 1  (U-ones; coincide con _chexpert_comparison_metrics)
        'negative'       : -1 → 0  (U-zeros)
        'ignore'         : -1 → mask=0 (excluido de loss y métricas)
    """
    cols_gt = [c for c in CHEXBERT_LABELS if c in df_chex.columns]
    if len(cols_gt) != NUM_LABELS:
        faltan = set(CHEXBERT_LABELS) - set(cols_gt)
        logger.warning(f"  GT CheXpert sin todas las columnas; faltan: {faltan}")

    df = df_split.copy()
    df["_sid"] = df["study_id"].apply(_norm_sid)
    chex = df_chex.copy()
    chex["_sid"] = chex["study_id"].apply(_norm_sid)
    chex = chex.drop_duplicates(subset=["_sid"])

    merged = df.merge(chex[["_sid"] + cols_gt], on="_sid", how="left")

    # Filas con GT disponible (al menos una columna no-NaN tras el merge)
    tiene_gt = merged[cols_gt].notna().any(axis=1)
    n_drop = int((~tiene_gt).sum())
    if n_drop:
        logger.info(f"  Filas sin GT CheXpert descartadas: {n_drop:,}/{len(merged):,}")
    df_ok = merged[tiene_gt].reset_index(drop=True)

    raw = df_ok[cols_gt].to_numpy(dtype=np.float64)   # con NaN
    nan_mask = np.isnan(raw)

    labels = np.zeros_like(raw, dtype=np.float32)
    masks = np.ones_like(raw, dtype=np.float32)

    pos = (raw == 1.0)
    unc = (raw == -1.0)
    labels[pos] = 1.0
    if uncertain == "positive":
        labels[unc] = 1.0
    elif uncertain == "negative":
        labels[unc] = 0.0
    elif uncertain == "ignore":
        masks[unc] = 0.0  # excluido
    else:
        raise ValueError(f"--uncertain inválido: {uncertain}")

    # NaN (no mencionado) → negativo (0) y SÍ cuenta (mask=1). labels ya es 0.
    labels[nan_mask] = 0.0

    return labels, masks, df_ok


# ─────────────────────────────────────────────────────────────────────────────
# 4. Dataset de clasificación (reutiliza índice + CLAHE del proyecto)
# ─────────────────────────────────────────────────────────────────────────────
class RadDinoCheXpertDataset(Dataset):
    """
    Devuelve (pixel_values, labels, mask) para cada imagen.

    • Imagen: estandarizar_imagen() del proyecto (CLAHE → RGB → resize) y luego
      normalización con las estadísticas NATIVAS de RadDINO (transform externo).
    • labels/mask: precomputados (construir_matriz_labels) → __getitem__ solo
      hace I/O de imagen (rápido, paralelizable en workers).
    """

    def __init__(
        self,
        df_ok: pd.DataFrame,
        labels: np.ndarray,
        masks: np.ndarray,
        image_index: Dict[str, Path],
        transform: transforms.Compose,
        image_size: Tuple[int, int],
        clahe_clip_limit: float,
        clahe_tile_grid_size: Tuple[int, int],
        augment_transform: Optional[transforms.Compose] = None,
    ):
        self.dicom_ids = df_ok["dicom_id"].astype(str).tolist()
        self.labels = torch.from_numpy(labels)
        self.masks = torch.from_numpy(masks)
        self.image_index = image_index
        self.transform = transform
        self.image_size = image_size
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.augment_transform = augment_transform

    def __len__(self) -> int:
        return len(self.dicom_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dicom_id = self.dicom_ids[idx]
        path = self.image_index.get(dicom_id)

        if path is None:
            pil = crear_imagen_placeholder(self.image_size)
        else:
            pil = estandarizar_imagen(
                path,
                target_size=self.image_size,
                clahe_clip_limit=self.clahe_clip_limit,
                clahe_tile_grid_size=self.clahe_tile_grid_size,
            )
            if pil is None:
                pil = crear_imagen_placeholder(self.image_size)

        if self.augment_transform is not None:
            pil = self.augment_transform(pil)

        pixel_values = self.transform(pil)   # (3, H, W) float32, normalizado RadDINO
        return {
            "pixel_values": pixel_values,
            "labels": self.labels[idx],
            "mask": self.masks[idx],
        }


def build_raddino_transform(image_proc, image_size: Tuple[int, int]) -> transforms.Compose:
    """ToTensor + Normalize con las estadísticas nativas de RadDINO."""
    mean = list(getattr(image_proc, "image_mean", [0.5307, 0.5307, 0.5307]))
    std = list(getattr(image_proc, "image_std", [0.2583, 0.2583, 0.2583]))
    # estandarizar_imagen ya entrega RGB en image_size; solo tensor + normalize.
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 5. Modelo — RadDINO (frozen | +LoRA) + cabezal lineal
# ─────────────────────────────────────────────────────────────────────────────
def _extract_cls(outputs) -> torch.Tensor:
    """Embedding denso [CLS]: pooler_output con fallback a last_hidden_state[:,0]."""
    pooled = getattr(outputs, "pooler_output", None)
    if pooled is not None:
        return pooled
    return outputs.last_hidden_state[:, 0]


class RadDinoClassifier(nn.Module):
    """
    Backbone RadDINO + cabezal nn.Linear(hidden → 14).

    apply_lora=False → backbone 100 % congelado (linear probing estricto).
    apply_lora=True  → LoRA en (query, value) del backbone + cabezal entrenable.
    El cabezal SIEMPRE opera en float32 (estabilidad de BCEWithLogits).
    """

    def __init__(
        self,
        apply_lora: bool = False,
        num_labels: int = NUM_LABELS,
        backbone_dtype: torch.dtype = torch.bfloat16,
        lora_r: int = RADDINO_LORA_R,
        lora_alpha: int = RADDINO_LORA_ALPHA,
        lora_dropout: float = RADDINO_LORA_DROPOUT,
        lora_targets: Optional[List[str]] = None,
    ):
        super().__init__()
        from transformers import AutoModel

        self.apply_lora = apply_lora
        self.frozen_backbone = not apply_lora

        logger.info(f"  Cargando backbone RadDINO ({RADDINO_MODEL_ID}, dtype={backbone_dtype})...")
        backbone = AutoModel.from_pretrained(RADDINO_MODEL_ID, torch_dtype=backbone_dtype)
        self.hidden_size = int(getattr(backbone.config, "hidden_size", 768))

        if apply_lora:
            from peft import LoraConfig, get_peft_model
            targets = lora_targets or RADDINO_TARGET_MODULES
            cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=None,                # passthrough puro de kwargs visuales
                target_modules=targets,
            )
            logger.info(f"  Inyectando LoRA en RadDINO (r={lora_r}, alpha={lora_alpha}, "
                        f"targets={targets})...")
            backbone = get_peft_model(backbone, cfg)
        else:
            for p in backbone.parameters():
                p.requires_grad = False

        self.backbone = backbone
        self.head = nn.Linear(self.hidden_size, num_labels)   # float32 por defecto

    # CLS del backbone (sin grad si está congelado)
    def forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.frozen_backbone:
            with torch.no_grad():
                out = self.backbone(pixel_values=pixel_values)
        else:
            out = self.backbone(pixel_values=pixel_values)
        return _extract_cls(out)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        cls = self.forward_features(pixel_values)
        return self.head(cls.float())

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        """Solo parámetros entrenables (cabezal [+ LoRA]) en CPU."""
        return {n: p.detach().cpu() for n, p in self.named_parameters() if p.requires_grad}


# ─────────────────────────────────────────────────────────────────────────────
# 6. Métricas multi-etiqueta (P/R/F1 por patología + macro/micro + AUROC)
# ─────────────────────────────────────────────────────────────────────────────
def compute_multilabel_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Devuelve (df_por_patologia, dict_resumen).

    Por patología: precision, recall, f1 (umbral) + AUROC (threshold-free) +
    soporte positivo/negativo. Macro = media sobre las 14 clases; micro = pooled.
    """
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        roc_auc_score = None

    preds = (probs >= threshold).astype(np.int32)
    rows: List[Dict] = []
    tp_tot = fp_tot = fn_tot = 0

    for c, name in enumerate(CHEXBERT_LABELS):
        m = masks[:, c].astype(bool)
        y = targets[m, c].astype(np.int32)
        p = preds[m, c]
        pr = probs[m, c]

        tp = int(((p == 1) & (y == 1)).sum())
        fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        tn = int(((p == 0) & (y == 0)).sum())
        tp_tot += tp; fp_tot += fp; fn_tot += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        auroc = float("nan")
        if roc_auc_score is not None and y.size > 0 and y.min() != y.max():
            try:
                auroc = float(roc_auc_score(y, pr))
            except Exception:
                auroc = float("nan")

        rows.append({
            "pathology": name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auroc": auroc,
            "support_pos": int(y.sum()),
            "support_neg": int((y == 0).sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    df_pat = pd.DataFrame(rows)

    micro_p = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else 0.0
    micro_r = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    resumen = {
        "macro_precision": float(df_pat["precision"].mean()),
        "macro_recall": float(df_pat["recall"].mean()),
        "macro_f1": float(df_pat["f1"].mean()),
        "macro_auroc": float(np.nanmean(df_pat["auroc"].to_numpy())),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
    }
    return df_pat, resumen


# ─────────────────────────────────────────────────────────────────────────────
# 7. Extracción de features [CLS] (cache para el linear probing)
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def extract_features(
    model: RadDinoClassifier,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    etiqueta: str = "",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extrae (features[N,H] fp16, labels[N,14], masks[N,14]) recorriendo el loader UNA vez."""
    from tqdm import tqdm
    model.eval()
    feats, labs, msks = [], [], []
    use_amp = amp_dtype is not None and device.type == "cuda"
    for batch in tqdm(loader, desc=f"  Extrayendo features {etiqueta}".rstrip(), unit="batch"):
        pv = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            cls = model.forward_features(pv)
        feats.append(cls.float().half().cpu())
        labs.append(batch["labels"])
        msks.append(batch["mask"])
    return torch.cat(feats), torch.cat(labs), torch.cat(msks)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Pérdida / pos_weight
# ─────────────────────────────────────────────────────────────────────────────
def compute_pos_weight(labels: np.ndarray, masks: np.ndarray, cap: float = 50.0) -> torch.Tensor:
    """pos_weight[c] = N_neg/N_pos por clase (con máscara). pos=0 → 1.0; recortado a 'cap'."""
    lab = labels * masks
    pos = lab.sum(axis=0)
    valid = masks.sum(axis=0)
    neg = np.clip(valid - pos, 0, None)
    pw = np.ones(labels.shape[1], dtype=np.float32)
    nz = pos > 0
    pw[nz] = np.clip(neg[nz] / np.clip(pos[nz], 1, None), 0.0, cap)
    return torch.tensor(pw, dtype=torch.float32)


def _masked_bce(criterion, logits: torch.Tensor, targets: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
    """BCEWithLogits con reducción manual respetando la máscara (clases ignoradas)."""
    per = criterion(logits, targets)            # (B,14), reduction='none'
    per = per * mask
    denom = mask.sum().clamp(min=1.0)
    return per.sum() / denom


# ─────────────────────────────────────────────────────────────────────────────
# 9. Evaluación (desde features cacheadas | desde loader end-to-end)
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def evaluate_from_features(
    head: nn.Module, feats: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor,
    criterion, device: torch.device, batch: int = 8192,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    head.eval()
    losses, probs = [], []
    n = feats.shape[0]
    for i in range(0, n, batch):
        f = feats[i:i + batch].to(device, non_blocking=True).float()
        y = labels[i:i + batch].to(device, non_blocking=True)
        m = masks[i:i + batch].to(device, non_blocking=True)
        logits = head(f)
        loss = _masked_bce(criterion, logits, y, m)
        losses.append(float(loss) * f.shape[0])
        probs.append(torch.sigmoid(logits).float().cpu())
    avg = sum(losses) / max(n, 1)
    return avg, torch.cat(probs).numpy(), labels.numpy(), masks.numpy()


@torch.inference_mode()
def evaluate_from_loader(
    model: RadDinoClassifier, loader: DataLoader, criterion,
    device: torch.device, amp_dtype: Optional[torch.dtype],
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    from tqdm import tqdm
    model.eval()
    use_amp = amp_dtype is not None and device.type == "cuda"
    losses, probs, labs, msks = [], [], [], []
    n = 0
    for batch in tqdm(loader, desc="  Validación", unit="batch", leave=False):
        pv = batch["pixel_values"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)
        m = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            logits = model(pv)
        loss = _masked_bce(criterion, logits.float(), y, m)
        bs = pv.shape[0]; n += bs
        losses.append(float(loss) * bs)
        probs.append(torch.sigmoid(logits.float()).cpu())
        labs.append(batch["labels"]); msks.append(batch["mask"])
    avg = sum(losses) / max(n, 1)
    return avg, torch.cat(probs).numpy(), torch.cat(labs).numpy(), torch.cat(msks).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 10. Construcción de splits (índice de imágenes compartido)
# ─────────────────────────────────────────────────────────────────────────────
def construir_split(
    nombre: str,
    csv_path: Path,
    image_index: Dict[str, Path],
    transform: transforms.Compose,
    df_chex: pd.DataFrame,
    args: argparse.Namespace,
    augment_transform: Optional[transforms.Compose] = None,
    num_limit: Optional[int] = None,
) -> Tuple[RadDinoCheXpertDataset, np.ndarray, np.ndarray]:
    logger.info(f"\n  [{nombre}] Cargando split: {csv_path.name}")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["dicom_id", "study_id"]).reset_index(drop=True)
    if num_limit:
        df = df.head(num_limit).copy()

    labels, masks, df_ok = construir_matriz_labels(df, df_chex, uncertain=args.uncertain)

    # Solo filas con imagen presente en el índice
    presentes = df_ok["dicom_id"].astype(str).map(lambda d: d in image_index)
    n_faltan = int((~presentes).sum())
    if n_faltan:
        logger.info(f"  [{nombre}] Sin imagen en el índice: {n_faltan:,} → descartadas")
        labels = labels[presentes.to_numpy()]
        masks = masks[presentes.to_numpy()]
        df_ok = df_ok[presentes.to_numpy()].reset_index(drop=True)

    ds = RadDinoCheXpertDataset(
        df_ok=df_ok, labels=labels, masks=masks,
        image_index=image_index, transform=transform,
        image_size=tuple(args.image_size_tuple),
        clahe_clip_limit=config.data.clahe_clip_limit,
        clahe_tile_grid_size=config.data.clahe_tile_grid_size,
        augment_transform=augment_transform,
    )
    pos_por_clase = (labels * masks).sum(axis=0).astype(int)
    logger.info(f"  [{nombre}] N={len(ds):,} | positivos/clase (min/máx)="
                f"{pos_por_clase.min()}/{pos_por_clase.max()}")
    return ds, labels, masks


def make_loader(ds: Dataset, batch_size: int, args: argparse.Namespace,
                shuffle: bool, drop_last: bool = False) -> DataLoader:
    nw = args.num_workers
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
        num_workers=nw, pin_memory=(args.device == "cuda"),
        persistent_workers=(nw > 0), prefetch_factor=(args.prefetch_factor if nw > 0 else None),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 11. Guardado de artefactos
# ─────────────────────────────────────────────────────────────────────────────
def guardar_curva(history: List[Dict], out_png: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        logger.info("  (matplotlib no disponible → se omite el PNG de la curva)")
        return
    if not history:
        return
    ep = [h["epoch"] for h in history]
    tr = [h["train_loss"] for h in history]
    vl = [h["val_loss"] for h in history]
    f1 = [h["val_macro_f1"] for h in history]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(ep, tr, "-o", color="#1f77b4", label="train_loss", markersize=3)
    ax1.plot(ep, vl, "-o", color="#d62728", label="val_loss", markersize=3)
    ax1.set_xlabel("Época"); ax1.set_ylabel("BCE Loss"); ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(ep, f1, "-s", color="#2ca02c", label="val_macro_f1", markersize=3)
    ax2.set_ylabel("val_macro_F1", color="#2ca02c")
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="best", fontsize=9)
    plt.title("RadDINO probe — curva de Loss + val macro-F1")
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
    logger.info(f"  ✓ Curva: {out_png}")


# ─────────────────────────────────────────────────────────────────────────────
# 12. CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sonda diagnóstica aislada de RadDINO (14 clases CheXpert).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Arquitectura
    p.add_argument("--apply-lora", action="store_true",
                   help="True: LoRA en RadDINO + cabezal. False (def.): linear probing estricto.")
    # Datos
    p.add_argument("--chexpert-csv", type=str, default=str(CHEXPERT_CSV_DEFAULT))
    p.add_argument("--image-size", type=int, default=int(config.data.image_size[0]),
                   help="Lado de la imagen cuadrada (224 = proyecto; 518 = nativo RadDINO).")
    p.add_argument("--uncertain", choices=["positive", "negative", "ignore"],
                   default="positive", help="Política para el GT incierto (-1).")
    p.add_argument("--augment", action="store_true",
                   help="Augmentación geométrica suave (±5° rot + ±5%% traslación) en train.")
    p.add_argument("--num-train", type=int, default=None, help="Limitar N de train (smoke).")
    p.add_argument("--num-val", type=int, default=None, help="Limitar N de val (smoke).")
    p.add_argument("--num-test", type=int, default=None, help="Limitar N de test (smoke).")
    p.add_argument("--eval-test", action="store_true",
                   help="Evaluar también el Test Set con el mejor modelo al final.")
    # Optimización / loader
    p.add_argument("--batch-size", type=int, default=128,
                   help="Batch del loader de imágenes (extracción / LoRA train).")
    p.add_argument("--head-batch-size", type=int, default=4096,
                   help="Batch del entrenamiento del cabezal sobre features cacheadas.")
    p.add_argument("--num-workers", type=int, default=min(32, os.cpu_count() or 8))
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--amp-dtype", choices=["bfloat16", "float32"], default="bfloat16")
    # Entrenamiento
    p.add_argument("--epochs", type=int, default=None,
                   help="Épocas (def.: 40 linear-cache / 8 LoRA).")
    p.add_argument("--lr", type=float, default=1e-3, help="LR del cabezal lineal.")
    p.add_argument("--lr-lora", type=float, default=5e-5, help="LR de los adaptadores LoRA.")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=None,
                   help="Early stopping sobre val_macro_f1 (def.: 8 linear / 4 LoRA).")
    p.add_argument("--threshold", type=float, default=0.5, help="Umbral de decisión.")
    p.add_argument("--no-pos-weight", action="store_true",
                   help="Desactivar pos_weight (por defecto se usa para el desbalance).")
    p.add_argument("--no-cache-features", action="store_true",
                   help="No cachear features en linear probing (fuerza end-to-end).")
    # Misc
    p.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR / "raddino_probe"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 13. Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    enable_perf_flags()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device.type
    args.image_size_tuple = (args.image_size, args.image_size)
    amp_dtype = torch.bfloat16 if (args.amp_dtype == "bfloat16" and device.type == "cuda") else None

    mode = "lora" if args.apply_lora else "linear"
    run_dir = Path(args.output_dir) / f"raddino_{mode}_probe"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Defaults dependientes del modo
    epochs = args.epochs if args.epochs is not None else (8 if args.apply_lora else 40)
    patience = args.patience if args.patience is not None else (4 if args.apply_lora else 8)
    cache_features = (not args.apply_lora) and (not args.no_cache_features) and (not args.augment)

    print("\n" + "=" * 78)
    print(f"  SONDA DIAGNÓSTICA AISLADA · RadDINO · modo={'LoRA' if args.apply_lora else 'LINEAR PROBING'}")
    print("=" * 78)
    print(f"  RadDINO        : {RADDINO_MODEL_ID}")
    print(f"  Clases CheXpert: {NUM_LABELS}")
    print(f"  image_size     : {args.image_size_tuple}  | CLAHE(clip={config.data.clahe_clip_limit},"
          f" grid={config.data.clahe_tile_grid_size})")
    print(f"  uncertain pol. : {args.uncertain}  | threshold={args.threshold}  | pos_weight="
          f"{not args.no_pos_weight}")
    print(f"  AMP dtype      : {args.amp_dtype}  | device={device.type}")
    print(f"  epochs         : {epochs}  | patience={patience}  | cache_features={cache_features}")
    print(f"  batch(img/head): {args.batch_size}/{args.head_batch_size}  | workers={args.num_workers}")
    print(f"  output         : {run_dir}")
    print("=" * 78 + "\n")

    if device.type != "cuda":
        logger.warning("CUDA no disponible: será MUY lento en CPU.")

    # ── Índice de imágenes (UNA sola construcción, reutilizando MimicCXRDataset) ──
    logger.info("Construyendo índice de imágenes (reutiliza MimicCXRDataset)...")
    processor = cargar_procesador(model_name=config.model.model_name)
    base_ds = MimicCXRDataset(
        csv_path=TRAIN_CSV, images_dir=IMAGES_DIR, processor=processor,
        image_size=tuple(args.image_size_tuple),
        clahe_clip_limit=config.data.clahe_clip_limit,
        clahe_tile_grid_size=config.data.clahe_tile_grid_size,
        prompt=None, augment=False,
    )
    image_index: Dict[str, Path] = base_ds._indice_imagenes
    logger.info(f"  Índice listo: {len(image_index):,} imágenes")

    # ── GT CheXpert + transform RadDINO ──
    chex_path = Path(args.chexpert_csv)
    if not chex_path.exists():
        gz = Path(str(chex_path) + ".gz")
        if gz.exists():
            chex_path = gz
    logger.info(f"Cargando GT CheXpert: {chex_path}")
    df_chex = pd.read_csv(chex_path, compression="infer")

    from transformers import AutoImageProcessor
    try:
        img_proc = AutoImageProcessor.from_pretrained(RADDINO_MODEL_ID)
    except Exception as exc:
        logger.warning(f"  AutoImageProcessor RadDINO no disponible ({exc}); uso stats DINOv2.")
        img_proc = None
    transform = build_raddino_transform(img_proc, tuple(args.image_size_tuple))

    aug_t = None
    if args.augment:
        aug_t = transforms.Compose([
            transforms.RandomRotation(5, interpolation=transforms.InterpolationMode.BILINEAR, fill=0),
            transforms.RandomAffine(0, translate=(0.05, 0.05),
                                    interpolation=transforms.InterpolationMode.BILINEAR, fill=0),
        ])

    # ── Datasets ──
    train_ds, train_labels, train_masks = construir_split(
        "train", TRAIN_CSV, image_index, transform, df_chex, args,
        augment_transform=aug_t, num_limit=args.num_train)
    val_ds, _, _ = construir_split(
        "val", VAL_CSV, image_index, transform, df_chex, args,
        augment_transform=None, num_limit=args.num_val)

    # ── Modelo ──
    # Frozen → bf16 (solo inferencia, ahorra VRAM). LoRA → fp32 + autocast bf16
    # (receta AMP estándar: pesos maestros fp32, cómputo bf16, sin GradScaler).
    backbone_dtype = torch.float32 if args.apply_lora else torch.bfloat16
    model = RadDinoClassifier(apply_lora=args.apply_lora, backbone_dtype=backbone_dtype)
    model.to(device)

    n_train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total_p = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parámetros entrenables: {n_train_p:,} / {n_total_p:,} "
                f"({100.0*n_train_p/max(n_total_p,1):.3f}%)")

    # ── pos_weight + criterio ──
    pos_weight = None
    if not args.no_pos_weight:
        pos_weight = compute_pos_weight(train_labels, train_masks).to(device)
        logger.info(f"  pos_weight (min/máx): {pos_weight.min():.2f}/{pos_weight.max():.2f}")
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    history: List[Dict] = []
    best_f1 = -1.0
    best_epoch = -1
    best_state = None
    t0 = time.time()

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║ RUTA A — LINEAR PROBING con CACHE de features (máxima eficiencia)       ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    if cache_features:
        logger.info(f"\n{SEP}\n  LINEAR PROBING · cache de features [CLS] (1 pasada por split)\n{SEP}")
        extract_bs = max(args.batch_size, 256)
        tr_loader = make_loader(train_ds, extract_bs, args, shuffle=False)
        va_loader = make_loader(val_ds, extract_bs, args, shuffle=False)

        f_tr, y_tr, m_tr = extract_features(model, tr_loader, device, amp_dtype, "[train]")
        f_va, y_va, m_va = extract_features(model, va_loader, device, amp_dtype, "[val]")
        del tr_loader, va_loader
        gc.collect(); torch.cuda.empty_cache()
        logger.info(f"  Features: train={tuple(f_tr.shape)}  val={tuple(f_va.shape)} (fp16)")

        head = model.head
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        steps_per_epoch = math.ceil(f_tr.shape[0] / args.head_batch_size)
        total_steps = steps_per_epoch * epochs
        warmup = max(1, int(total_steps * args.warmup_ratio))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s / warmup) if s < warmup else
            0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))))

        n = f_tr.shape[0]
        for epoch in range(1, epochs + 1):
            head.train()
            perm = torch.randperm(n)
            running = 0.0
            for i in range(0, n, args.head_batch_size):
                idx = perm[i:i + args.head_batch_size]
                f = f_tr[idx].to(device, non_blocking=True).float()
                y = y_tr[idx].to(device, non_blocking=True)
                m = m_tr[idx].to(device, non_blocking=True)
                logits = head(f)
                loss = _masked_bce(criterion, logits, y, m)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step(); sched.step()
                running += float(loss) * f.shape[0]
            train_loss = running / max(n, 1)

            val_loss, probs, tgts, msks = evaluate_from_features(
                head, f_va, y_va, m_va, criterion, device, batch=args.head_batch_size)
            _, resumen = compute_multilabel_metrics(probs, tgts, msks, args.threshold)

            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                   "val_macro_precision": resumen["macro_precision"],
                   "val_macro_recall": resumen["macro_recall"],
                   "val_macro_f1": resumen["macro_f1"],
                   "val_macro_auroc": resumen["macro_auroc"],
                   "val_micro_f1": resumen["micro_f1"],
                   "lr": opt.param_groups[0]["lr"]}
            history.append(row)
            logger.info(f"  [E{epoch:02d}/{epochs}] train={train_loss:.4f} val={val_loss:.4f} "
                        f"| macroF1={resumen['macro_f1']:.4f} microF1={resumen['micro_f1']:.4f} "
                        f"AUROC={resumen['macro_auroc']:.4f}")
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

            if resumen["macro_f1"] > best_f1:
                best_f1 = resumen["macro_f1"]; best_epoch = epoch
                best_state = model.trainable_state_dict()
                torch.save(best_state, run_dir / "best_state.pt")
            elif epoch - best_epoch >= patience:
                logger.info(f"  Early stopping (sin mejora en {patience} épocas).")
                break

        # Mejor cabezal para la evaluación final
        if best_state is not None:
            model.load_state_dict(best_state, strict=False)
        val_loss, probs, tgts, msks = evaluate_from_features(
            model.head, f_va, y_va, m_va, criterion, device, batch=args.head_batch_size)
        df_pat_val, resumen_val = compute_multilabel_metrics(probs, tgts, msks, args.threshold)
        del f_tr, y_tr, m_tr, f_va, y_va, m_va
        gc.collect(); torch.cuda.empty_cache()

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║ RUTA B — END-TO-END (LoRA, o linear con --no-cache-features/--augment)  ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    else:
        logger.info(f"\n{SEP}\n  ENTRENAMIENTO END-TO-END "
                    f"({'LoRA + cabezal' if args.apply_lora else 'cabezal sobre backbone congelado'})\n{SEP}")
        tr_loader = make_loader(train_ds, args.batch_size, args, shuffle=True, drop_last=True)
        va_loader = make_loader(val_ds, max(args.batch_size, 256), args, shuffle=False)

        # Grupos de parámetros: LoRA (lr_lora) + cabezal (lr)
        head_params = list(model.head.parameters())
        lora_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith("head.")]
        groups = [{"params": head_params, "lr": args.lr}]
        if lora_params:
            groups.append({"params": lora_params, "lr": args.lr_lora})
        opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

        total_steps = len(tr_loader) * epochs
        warmup = max(1, int(total_steps * args.warmup_ratio))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s / warmup) if s < warmup else
            0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))))
        use_amp = amp_dtype is not None and device.type == "cuda"

        from tqdm import tqdm
        for epoch in range(1, epochs + 1):
            model.train()
            running = 0.0; seen = 0
            for batch in tqdm(tr_loader, desc=f"  Train E{epoch:02d}/{epochs}", unit="batch"):
                pv = batch["pixel_values"].to(device, non_blocking=True)
                y = batch["labels"].to(device, non_blocking=True)
                m = batch["mask"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    logits = model(pv)
                loss = _masked_bce(criterion, logits.float(), y, m)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step(); sched.step()
                bs = pv.shape[0]; running += float(loss) * bs; seen += bs
            train_loss = running / max(seen, 1)

            val_loss, probs, tgts, msks = evaluate_from_loader(
                model, va_loader, criterion, device, amp_dtype)
            _, resumen = compute_multilabel_metrics(probs, tgts, msks, args.threshold)

            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                   "val_macro_precision": resumen["macro_precision"],
                   "val_macro_recall": resumen["macro_recall"],
                   "val_macro_f1": resumen["macro_f1"],
                   "val_macro_auroc": resumen["macro_auroc"],
                   "val_micro_f1": resumen["micro_f1"],
                   "lr": opt.param_groups[0]["lr"]}
            history.append(row)
            logger.info(f"  [E{epoch:02d}/{epochs}] train={train_loss:.4f} val={val_loss:.4f} "
                        f"| macroF1={resumen['macro_f1']:.4f} microF1={resumen['micro_f1']:.4f} "
                        f"AUROC={resumen['macro_auroc']:.4f}")
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

            if resumen["macro_f1"] > best_f1:
                best_f1 = resumen["macro_f1"]; best_epoch = epoch
                best_state = model.trainable_state_dict()
                torch.save(best_state, run_dir / "best_state.pt")
            elif epoch - best_epoch >= patience:
                logger.info(f"  Early stopping (sin mejora en {patience} épocas).")
                break

        if best_state is not None:
            model.load_state_dict(best_state, strict=False)
        val_loss, probs, tgts, msks = evaluate_from_loader(model, va_loader, criterion, device, amp_dtype)
        df_pat_val, resumen_val = compute_multilabel_metrics(probs, tgts, msks, args.threshold)
        del tr_loader, va_loader
        gc.collect(); torch.cuda.empty_cache()

    # ── Métricas por patología (val, mejor modelo) ──
    df_pat_val.insert(0, "split", "val")
    per_pathology_frames = [df_pat_val]

    print(f"\n{'=' * 78}\n  MÉTRICAS · VAL (mejor época = {best_epoch})\n{'=' * 78}")
    print(df_pat_val[["pathology", "precision", "recall", "f1", "auroc",
                      "support_pos"]].to_string(index=False))
    print(f"\n  MACRO  → P={resumen_val['macro_precision']:.4f}  R={resumen_val['macro_recall']:.4f}  "
          f"F1={resumen_val['macro_f1']:.4f}  AUROC={resumen_val['macro_auroc']:.4f}")
    print(f"  MICRO  → P={resumen_val['micro_precision']:.4f}  R={resumen_val['micro_recall']:.4f}  "
          f"F1={resumen_val['micro_f1']:.4f}")

    # ── Test (opcional) ──
    resumen_test = None
    if args.eval_test:
        logger.info(f"\n{SEP}\n  EVALUACIÓN FINAL · TEST SET\n{SEP}")
        test_ds, _, _ = construir_split(
            "test", TEST_CSV, image_index, transform, df_chex, args,
            augment_transform=None, num_limit=args.num_test)
        te_loader = make_loader(test_ds, max(args.batch_size, 256), args, shuffle=False)
        _, probs_t, tgts_t, msks_t = evaluate_from_loader(model, te_loader, criterion, device, amp_dtype)
        df_pat_test, resumen_test = compute_multilabel_metrics(probs_t, tgts_t, msks_t, args.threshold)
        df_pat_test.insert(0, "split", "test")
        per_pathology_frames.append(df_pat_test)
        print(f"\n  TEST MACRO → P={resumen_test['macro_precision']:.4f}  "
              f"R={resumen_test['macro_recall']:.4f}  F1={resumen_test['macro_f1']:.4f}  "
              f"AUROC={resumen_test['macro_auroc']:.4f}  | microF1={resumen_test['micro_f1']:.4f}")
        del te_loader; gc.collect(); torch.cuda.empty_cache()

    # ── Persistir artefactos ──
    pd.concat(per_pathology_frames, ignore_index=True).to_csv(
        run_dir / "metrics_per_pathology.csv", index=False)
    guardar_curva(history, run_dir / "loss_curve.png")

    summary = {
        "mode": "lora_probe" if args.apply_lora else "linear_probe",
        "apply_lora": bool(args.apply_lora),
        "raddino_model": RADDINO_MODEL_ID,
        "num_labels": NUM_LABELS,
        "labels": CHEXBERT_LABELS,
        "image_size": list(args.image_size_tuple),
        "uncertain_policy": args.uncertain,
        "pos_weight": (not args.no_pos_weight),
        "threshold": args.threshold,
        "amp_dtype": args.amp_dtype,
        "cache_features": cache_features,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "trainable_params": int(n_train_p),
        "total_params": int(n_total_p),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": (len(test_ds) if args.eval_test else None),
        "val_metrics": resumen_val,
        "test_metrics": resumen_test,
        "lora": ({"r": RADDINO_LORA_R, "alpha": RADDINO_LORA_ALPHA,
                  "dropout": RADDINO_LORA_DROPOUT, "targets": RADDINO_TARGET_MODULES}
                 if args.apply_lora else None),
        "elapsed_min": round((time.time() - t0) / 60.0, 2),
        "device": device.type,
    }
    (run_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'=' * 78}\n  COMPLETADO ({summary['elapsed_min']} min)\n{'=' * 78}")
    print(f"  history          : {run_dir / 'history.csv'}")
    print(f"  per_pathology    : {run_dir / 'metrics_per_pathology.csv'}")
    print(f"  summary          : {run_dir / 'metrics_summary.json'}")
    print(f"  best_state       : {run_dir / 'best_state.pt'}")
    print(f"  loss_curve       : {run_dir / 'loss_curve.png'}")
    print("=" * 78 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
