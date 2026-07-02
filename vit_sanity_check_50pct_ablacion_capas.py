#!/usr/bin/env python3
"""
vit_sanity_check_50pct_ablacion_capas.py — Ablación de Capas Intermedias (BLIP-2 · 50 %)
=======================================================================================
Sanity Check + Estudio de Ablación de Capas Intermedias del ViT congelado.

¿Contienen las capas intermedias (penúltima, antepenúltima…) mejores
características radiológicas que la última capa? Este script extrae el token
[CLS] de una capa arbitraria del ViT (--layer) y entrena un clasificador
lineal sobre esas representaciones para comparar su poder discriminativo.

Versión adaptada al modelo entrenado con el 50 % de MIMIC-CXR (Config_B1):
    · ViT LoRA  → phase2_results-50pct/checkpoints/best_model_Config_B1/vit
    · Imágenes  → imagenes_50_porciento
    · Splits    → train_split_50pct.csv / val_split_50pct.csv

Estrategia
----------
1. Carga el ViT de BLIP-2 base.
2. Inyecta el adaptador LoRA de Fase 2 (best_model_Config_B1/vit) y fusiona
   con merge_and_unload().
3. Congela el 100% del ViT (requires_grad=False).
4. Conecta un clasificador lineal de 14 clases CheXpert.
5. Entrena SOLO el clasificador (Linear Probing) con BCEWithLogitsLoss.
6. Evalúa por época: Loss, AUC-ROC macro y F1-macro.

Etiquetas CheXpert (14 clases)
-------------------------------
Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum,
Fracture, Lung Lesion, Lung Opacity, No Finding, Pleural Effusion,
Pleural Other, Pneumonia, Pneumothorax, Support Devices

Preprocesamiento
----------------
Las imágenes pasan por MimicCXRDataset + Blip2Processor: CLAHE, resize 224,
normalización ImageNet — exactamente igual que en el entrenamiento BLIP-2.

Uso
---
    python vit_sanity_check_50pct_ablacion_capas.py                    # ultimas 10 capas (-1..-10)
    python vit_sanity_check_50pct_ablacion_capas.py --num-layers 6     # ultimas 6 capas
    python vit_sanity_check_50pct_ablacion_capas.py --layers -1 -3 -5  # capas concretas
    python vit_sanity_check_50pct_ablacion_capas.py --max-train-samples 0  # dataset COMPLETO (133K)
    python vit_sanity_check_50pct_ablacion_capas.py --no-phase2            # ViT base sin Fase 2

Salida (un unico lugar)
-----------------------
    ablacion_capas_results/
        layer_-1/  history_vit_sanity_check.csv  +  best_linear_classifier.pt
        layer_-2/  ...
        ...
        resumen_ablacion_capas.csv   <- UNA fila por capa con su mejor AUC-ROC, F1, etc.

Nota sobre la cantidad de imágenes
----------------------------------
El split 50 % tiene 133 290 train / 38 989 val imágenes (vs ~26 K en el 10 %).
Para un sanity check ágil se submuestrea por defecto a 20 000 train / 5 000 val
(--max-train-samples / --max-val-samples; usa 0 para procesar el dataset completo).

Autor : Evelyn Silva Rozas
Fecha : Junio 2026
"""

import argparse
import gc
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
cv2.setNumThreads(0)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm

# ── Módulos del proyecto ──────────────────────────────────────────────────────
from config import config
from model_utils import cargar_procesador
from data_loader import MimicCXRDataset

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR    = config.paths.base_dir
IMAGES_DIR  = BASE_DIR / "imagenes_50_porciento"

TRAIN_CSV   = BASE_DIR / "train_split_50pct.csv"
VAL_CSV     = BASE_DIR / "val_split_50pct.csv"

CHEXPERT_CSV = BASE_DIR / "mimic-cxr-2.0.0-chexpert.csv"

VIT_LORA_DIR = (
    BASE_DIR
    / "phase2_results-50pct"
    / "checkpoints"
    / "best_model_Config_B1"
    / "vit"
)

# Directorio UNICO para todo el estudio de ablacion de capas.
# Dentro se crea una subcarpeta por capa (layer_-1/, layer_-2/, ...) con su
# history + best_linear_classifier.pt, y un CSV resumen con el MEJOR valor de
# cada capa — todo en un solo lugar.
OUTPUT_DIR = BASE_DIR / "ablacion_capas_results"

# CSV resumen: UNA fila por capa con su mejor AUC-ROC, F1-macro, etc.
ABLATION_SUMMARY_CSV = OUTPUT_DIR / "resumen_ablacion_capas.csv"

# 14 etiquetas CheXpert (orden original del CSV)
CHEXPERT_LABELS: List[str] = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax", "Support Devices",
]
N_CLASSES = len(CHEXPERT_LABELS)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATASET CON ETIQUETAS CHEXPERT
# ─────────────────────────────────────────────────────────────────────────────

class CheXpertLabeledDataset(Dataset):
    """
    Wrapper sobre MimicCXRDataset que añade las etiquetas multi-label CheXpert.

    Estrategia de etiquetas
    -----------------------
    - 1.0  → positivo
    - -1.0 → incierto → se trata como positivo (política U-ones, estándar CheXpert)
    - 0.0 / NaN → negativo → 0

    Solo se incluyen muestras que tienen study_id en el CSV de CheXpert.
    """

    def __init__(
        self,
        base_dataset: MimicCXRDataset,
        chexpert_df:  pd.DataFrame,
    ):
        # Índices del base_dataset que tienen etiquetas CheXpert disponibles
        base_df = base_dataset.data.copy()
        base_df["_idx"] = range(len(base_df))

        # Join por study_id (ambos son strings)
        base_df["study_id"] = base_df["study_id"].astype(str)
        chexpert_df = chexpert_df.copy()
        chexpert_df["study_id"] = chexpert_df["study_id"].astype(str)

        merged = base_df.merge(
            chexpert_df[["study_id"] + CHEXPERT_LABELS],
            on="study_id",
            how="inner",
        )
        merged = merged.dropna(subset=["_idx"])

        self.base_dataset = base_dataset
        self.indices      = merged["_idx"].tolist()
        self.labels       = merged[CHEXPERT_LABELS].fillna(0).clip(lower=0).values.astype(np.float32)
        # U-ones: -1 → 1
        # clip(lower=0) ya convierte -1 → 0; necesitamos U-ones explícito:
        raw = merged[CHEXPERT_LABELS].fillna(0).values.astype(np.float32)
        self.labels = np.where(raw == -1.0, 1.0, np.where(raw == 1.0, 1.0, 0.0)).astype(np.float32)

        logger.info(
            f"   CheXpertLabeledDataset: {len(self.indices):,} muestras con "
            f"etiquetas ({len(base_df) - len(self.indices):,} sin match descartadas)"
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample       = self.base_dataset[self.indices[idx]]
        pixel_values = sample["pixel_values"]           # [3, H, W] bfloat16/float32
        labels       = torch.tensor(self.labels[idx], dtype=torch.float32)
        return pixel_values, labels


# ─────────────────────────────────────────────────────────────────────────────
# 2.  ARQUITECTURA: ViT + CLASIFICADOR LINEAL
# ─────────────────────────────────────────────────────────────────────────────

class ViTLinearClassifier(nn.Module):
    """
    ViT (EVA-CLIP de BLIP-2) 100% congelado + capa lineal de 14 clases.

    Forward
    -------
    pixel_values → ViT → hidden_states[layer_idx] → CLS token [:, 0, :] → Linear → logits

    Estudio de ablación de capas: layer_idx selecciona de qué capa oculta del
    ViT se extrae el token [CLS].
      -1 = última capa (default)   -2 = penúltima   -3 = antepenúltima ...

    hidden_states es una tupla de (n_layers + 1) tensores; el índice 0 son los
    embeddings de entrada, por lo que la indexación negativa apunta a las
    últimas capas del encoder de forma intuitiva.
    """

    def __init__(self, vit: nn.Module, hidden_size: int, layer_idx: int = -1, n_classes: int = N_CLASSES):
        super().__init__()
        self.vit        = vit
        self.layer_idx  = layer_idx
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [B, 3, H, W]
        outputs = self.vit(pixel_values=pixel_values, return_dict=True, output_hidden_states=True)

        # Capa deseada de la tupla de hidden_states → token [CLS] de esa capa
        target_layer = outputs.hidden_states[self.layer_idx]   # [B, seq_len, hidden_size]
        features     = target_layer[:, 0, :]                   # CLS token [B, hidden_size]

        # Cast a float32 para la capa lineal (el ViT puede estar en bfloat16)
        features = features.float()
        return self.classifier(features)                       # [B, 14]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FEATURE CACHE — extracción única de embeddings ViT
# ─────────────────────────────────────────────────────────────────────────────

class FeatureDataset(Dataset):
    """
    Dataset ultra-ligero que sirve embeddings pre-extraídos del ViT.
    Una vez construido, el ViT nunca vuelve a ejecutarse durante el entrenamiento.
    """
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features   # [N, hidden_size] float32 en CPU
        self.labels   = labels     # [N, 14]          float32 en CPU

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


@torch.inference_mode()
def precompute_features(
    vit:           nn.Module,
    dataset:       CheXpertLabeledDataset,
    batch_size:    int,
    num_workers:   int,
    device:        torch.device,
    layer_indices: List[int],
    split_name:    str = "",
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
    """
    Pasa TODO el dataset UNA sola vez por el ViT congelado y extrae el token
    [CLS] de CADA capa en `layer_indices` simultáneamente.

    El forward del ViT (carga de imagen + CLAHE + encoder) es el cuello de
    botella; con output_hidden_states=True obtenemos las N capas en una única
    pasada, en vez de releer el dataset N veces (clave para el barrido).

    Returns
    -------
    (features_por_capa, labels)
        features_por_capa : dict {layer_idx: [N, hidden_size] float32 CPU}
        labels            : [N, 14] float32 CPU
    """
    loader = DataLoader(
        dataset,
        batch_size  = batch_size * 2,   # sin grad → batch más grande
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )

    feats_por_capa: Dict[int, List[torch.Tensor]] = {L: [] for L in layer_indices}
    all_labels: List[torch.Tensor] = []

    vit.eval()
    for pixel_values, labels in tqdm(
        loader, desc=f"  Pre-extrayendo features [{split_name}] ({len(layer_indices)} capas)"
    ):
        pixel_values = pixel_values.to(device)
        outputs = vit(pixel_values=pixel_values, return_dict=True, output_hidden_states=True)
        hidden_states = outputs.hidden_states   # tupla (n_layers + 1) de [B, seq, hidden]

        # token [CLS] de cada capa pedida (misma pasada para todas)
        for L in layer_indices:
            feats_por_capa[L].append(hidden_states[L][:, 0, :].float().cpu())
        all_labels.append(labels.cpu())

    labels_cat = torch.cat(all_labels, dim=0)        # [N, 14]
    features   = {L: torch.cat(feats_por_capa[L], dim=0) for L in layer_indices}
    ref_shape  = tuple(next(iter(features.values())).shape)
    logger.info(
        f"  Features [{split_name}]: {len(layer_indices)} capas × {ref_shape}  "
        f"labels: {tuple(labels_cat.shape)}  dtype=torch.float32"
    )
    return features, labels_cat


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CARGA DEL VIT CON PESOS DE FASE 2
# ─────────────────────────────────────────────────────────────────────────────

def load_vit_phase2(vit_lora_dir: Path, device: torch.device) -> Tuple[nn.Module, int]:
    """
    Carga el ViT de BLIP-2, inyecta el adaptador LoRA de Fase 2, fusiona y congela.

    Returns
    -------
    (vit_model, hidden_size)
    """
    from transformers import Blip2ForConditionalGeneration
    from peft import PeftModel

    logger.info("  Cargando BLIP-2 completo para extraer ViT...")
    full_model = Blip2ForConditionalGeneration.from_pretrained(
        config.model.model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",    # carga en CPU primero para extraer solo el ViT
    )

    # Extraer ViT y descartar el resto (liberar RAM/VRAM)
    vit = full_model.vision_model
    hidden_size = vit.config.hidden_size
    logger.info(f"  ViT extraído: hidden_size={hidden_size}")

    del full_model.language_model
    del full_model.qformer
    del full_model.language_projection
    del full_model
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("  Resto del modelo descartado (RAM liberada)")

    if vit_lora_dir.exists():
        logger.info(f"  Cargando ViT LoRA de Fase 2 desde {vit_lora_dir}...")
        vit_peft = PeftModel.from_pretrained(vit, str(vit_lora_dir), is_trainable=False)
        vit = vit_peft.merge_and_unload()
        logger.info("  LoRA fusionado (merge_and_unload) ✓")
    else:
        logger.warning(f"  ViT LoRA dir no encontrado: {vit_lora_dir}")
        logger.warning("  Continuando con ViT base (sin pesos de Fase 2)")

    # Congelar TODO el ViT
    for param in vit.parameters():
        param.requires_grad = False
    n_frozen = sum(p.numel() for p in vit.parameters())
    logger.info(f"  ViT 100% congelado: {n_frozen:,} parámetros (requires_grad=False)")

    vit = vit.to(device=device, dtype=torch.bfloat16)
    return vit, hidden_size


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    all_labels: np.ndarray,
    all_probs:  np.ndarray,
    threshold:  float = 0.5,
) -> Tuple[float, float]:
    """
    Calcula AUC-ROC macro y F1-macro para clasificación multi-label.

    Returns
    -------
    (auc_macro, f1_macro)
    """
    # AUC-ROC — solo para clases con al menos 2 clases presentes en labels
    auc_scores = []
    for i in range(all_labels.shape[1]):
        col = all_labels[:, i]
        if len(np.unique(col)) < 2:
            continue
        try:
            auc_scores.append(roc_auc_score(col, all_probs[:, i]))
        except Exception:
            pass
    auc_macro = float(np.mean(auc_scores)) if auc_scores else float("nan")

    # F1-macro
    preds = (all_probs >= threshold).astype(int)
    f1_macro = f1_score(all_labels, preds, average="macro", zero_division=0)

    return auc_macro, f1_macro


# ─────────────────────────────────────────────────────────────────────────────
# 6.  BUCLE DE ENTRENAMIENTO (sobre features cacheadas)
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model:        nn.Linear,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    epochs:       int,
    lr:           float,
    device:       torch.device,
    output_dir:   Path,
) -> pd.DataFrame:
    """
    Entrena el clasificador lineal sobre features pre-cacheadas.
    El ViT ya no forma parte del grafo — cada época es solo una capa lineal.

    Returns
    -------
    DataFrame con historial: epoch, train_loss, val_loss, auc_roc_macro, f1_macro
    """
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    n_classifier = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parámetros entrenables (clasificador lineal): {n_classifier:,}")

    history = []
    best_auc  = -1.0
    best_path = output_dir / "best_linear_classifier.pt"

    for epoch in range(1, epochs + 1):
        # ── TRAIN ─────────────────────────────────────────────────────────────
        model.train()
        train_losses = []

        for features, labels in tqdm(
            train_loader, desc=f"  Epoch {epoch}/{epochs} [train]", leave=False
        ):
            features = features.to(device)
            labels   = labels.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))

        # ── VAL ───────────────────────────────────────────────────────────────
        model.eval()
        val_losses   = []
        all_labels   = []
        all_probs    = []

        with torch.no_grad():
            for features, labels in tqdm(
                val_loader, desc=f"  Epoch {epoch}/{epochs} [val]  ", leave=False
            ):
                features = features.to(device)
                labels   = labels.to(device)
                logits   = model(features)
                loss     = criterion(logits, labels)
                val_losses.append(loss.item())

                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels.cpu().numpy())

        val_loss  = float(np.mean(val_losses))
        all_labels = np.concatenate(all_labels, axis=0)
        all_probs  = np.concatenate(all_probs,  axis=0)

        auc_macro, f1_macro = compute_metrics(all_labels, all_probs)

        # Guardar mejor modelo
        if not np.isnan(auc_macro) and auc_macro > best_auc:
            best_auc = auc_macro
            torch.save(model.state_dict(), best_path)
            marker = " ← best"
        else:
            marker = ""

        logger.info(
            f"  Epoch {epoch:>2}/{epochs} | "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"AUC-ROC={auc_macro:.4f}  F1-macro={f1_macro:.4f}{marker}"
        )

        history.append({
            "epoch":        epoch,
            "train_loss":   train_loss,
            "val_loss":     val_loss,
            "auc_roc_macro": auc_macro,
            "f1_macro":     f1_macro,
        })

    df_history = pd.DataFrame(history)
    df_history.to_csv(output_dir / "history_vit_sanity_check.csv", index=False)
    logger.info(f"\n  Historia guardada → {output_dir / 'history_vit_sanity_check.csv'}")
    logger.info(f"  Mejor clasificador → {best_path}  (AUC-ROC={best_auc:.4f})")

    return df_history


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Linear Probing sobre el ViT de Fase 2 (Sanity Check CheXpert 14 clases)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs",     type=int,   default=10,   help="Épocas de entrenamiento")
    parser.add_argument("--batch-size", type=int,   default=64,   help="Batch size")
    parser.add_argument("--lr",         type=float, default=1e-3, help="Learning rate (solo para el clasificador)")
    parser.add_argument("--num-workers",type=int,   default=8,    help="DataLoader workers (50pct: 172K imgs)")
    parser.add_argument("--max-train-samples", type=int, default=20000,
                        help="Submuestreo de TRAIN para un sanity check ágil (0 = completo, 133K)")
    parser.add_argument("--max-val-samples",   type=int, default=5000,
                        help="Submuestreo de VAL para un sanity check ágil (0 = completo, 39K)")
    parser.add_argument("--seed",       type=int,   default=42,   help="Semilla del submuestreo")
    parser.add_argument("--no-phase2",  action="store_true",      help="Usar ViT base sin pesos de Fase 2 (ablación)")
    parser.add_argument(
        "--num-layers", type=int, default=10,
        help="Numero de capas FINALES del ViT a evaluar: -1, -2, ..., -N (default 10).",
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=None,
        help="Lista explicita de capas a evaluar (ej. --layers -1 -3 -5). "
             "Si se indica, sobreescribe --num-layers.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # ── Capas a evaluar ───────────────────────────────────────────────────────────
    # Por defecto, las últimas N capas: [-1, -2, …, -num_layers].
    # --layers permite una lista explícita (ej. -1 -3 -5).
    if args.layers:
        layer_indices = list(args.layers)
    else:
        layer_indices = [-(i + 1) for i in range(args.num_layers)]

    # Directorio UNICO para todo el estudio → subcarpeta por capa + CSV resumen
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\n{'='*65}")
    logger.info("  VIT — Ablación de Capas Intermedias (Linear Probing CheXpert · 50 %)")
    logger.info(f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}")
    logger.info(f"  capas a evaluar ({len(layer_indices)}): {layer_indices}")
    logger.info(f"  device={device}  no_phase2={args.no_phase2}")
    logger.info(f"  max_train={args.max_train_samples or 'ALL'}  "
                f"max_val={args.max_val_samples or 'ALL'}  seed={args.seed}")
    logger.info(f"  output_dir={OUTPUT_DIR.name}/  (subcarpeta por capa + resumen CSV)")
    logger.info(f"{'='*65}\n")

    # ── Cargar etiquetas CheXpert ─────────────────────────────────────────────
    if not CHEXPERT_CSV.exists():
        logger.error(f"CSV CheXpert no encontrado: {CHEXPERT_CSV}")
        return 1
    logger.info(f"  Cargando etiquetas CheXpert desde {CHEXPERT_CSV.name}...")
    chexpert_df = pd.read_csv(CHEXPERT_CSV)
    logger.info(f"  {len(chexpert_df):,} estudios con etiquetas CheXpert")

    # ── Processor y datasets base ─────────────────────────────────────────────
    logger.info("  Cargando Blip2Processor...")
    processor = cargar_procesador(model_name=config.model.model_name)

    shared_kwargs = dict(
        images_dir           = IMAGES_DIR,
        processor            = processor,
        image_size           = config.data.image_size,
        clahe_clip_limit     = config.data.clahe_clip_limit,
        clahe_tile_grid_size = config.data.clahe_tile_grid_size,
        max_length           = 512,
        padding              = config.data.padding,
        prompt               = "",   # sin prompt; solo necesitamos pixel_values
    )

    logger.info("  Cargando Train Dataset (50%)...")
    base_train = MimicCXRDataset(csv_path=TRAIN_CSV, **shared_kwargs)
    logger.info("  Cargando Val Dataset   (50%)...")
    base_val   = MimicCXRDataset(csv_path=VAL_CSV,   **shared_kwargs)

    # ── Submuestreo opcional (la versión 50 % tiene 133K train / 39K val) ─────
    # Un sanity check no necesita las 133K imágenes: un subconjunto aleatorio
    # da una señal AUC-ROC fiable en una fracción del tiempo de extracción de
    # features.  --max-*-samples 0 procesa el dataset completo.
    if args.max_train_samples and len(base_train.data) > args.max_train_samples:
        base_train.data = base_train.data.sample(
            n=args.max_train_samples, random_state=args.seed).reset_index(drop=True)
        logger.info(f"  [submuestreo] Train → {len(base_train.data):,} (seed={args.seed})")
    if args.max_val_samples and len(base_val.data) > args.max_val_samples:
        base_val.data = base_val.data.sample(
            n=args.max_val_samples, random_state=args.seed).reset_index(drop=True)
        logger.info(f"  [submuestreo] Val   → {len(base_val.data):,} (seed={args.seed})")

    # ── Datasets con etiquetas CheXpert ───────────────────────────────────────
    logger.info("  Construyendo datasets con etiquetas CheXpert...")
    train_ds = CheXpertLabeledDataset(base_train, chexpert_df)
    val_ds   = CheXpertLabeledDataset(base_val,   chexpert_df)

    if len(train_ds) == 0 or len(val_ds) == 0:
        logger.error("No hay muestras con etiquetas CheXpert. Verifica el CSV.")
        return 1

    logger.info(f"  Train: {len(train_ds):,}  |  Val: {len(val_ds):,}\n")

    # ── Cargar ViT ────────────────────────────────────────────────────────────
    vit_dir = VIT_LORA_DIR if not args.no_phase2 else Path("__nonexistent__")
    vit, hidden_size = load_vit_phase2(vit_dir, device)

    # ── PRE-EXTRAER FEATURES — UNA sola pasada por el ViT, TODAS las capas ────
    # El forward del ViT (carga de imagen + CLAHE + encoder) es el cuello de
    # botella.  Con output_hidden_states=True extraemos el token [CLS] de las N
    # capas en una ÚNICA pasada, en vez de releer el dataset N veces.
    logger.info(
        f"\n  Pre-extrayendo embeddings del ViT "
        f"(UNA pasada, {len(layer_indices)} capas)..."
    )
    train_feats, train_lbls = precompute_features(
        vit, train_ds, args.batch_size, args.num_workers, device,
        layer_indices=layer_indices, split_name="train",
    )
    val_feats, val_lbls = precompute_features(
        vit, val_ds, args.batch_size, args.num_workers, device,
        layer_indices=layer_indices, split_name="val",
    )

    # ViT ya no es necesario en GPU → liberar VRAM
    del vit
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("  ViT descargado de VRAM tras extracción de features\n")

    # ── BARRIDO: un clasificador lineal por capa (sobre features cacheadas) ───
    # El ViT ya no está en el grafo; cada capa entrena solo un nn.Linear → rápido.
    resumen_rows = []
    for L in layer_indices:
        logger.info(f"\n{'─'*65}")
        logger.info(f"  CAPA {L}  →  entrenando clasificador lineal ({hidden_size}→{N_CLASSES})")
        logger.info(f"{'─'*65}")

        layer_dir = OUTPUT_DIR / f"layer_{L}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        train_loader = DataLoader(
            FeatureDataset(train_feats[L], train_lbls),
            batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True,
        )
        val_loader = DataLoader(
            FeatureDataset(val_feats[L], val_lbls),
            batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True,
        )

        clf_model = nn.Linear(hidden_size, N_CLASSES).to(device)
        df_history = train(
            model        = clf_model,
            train_loader = train_loader,
            val_loader   = val_loader,
            epochs       = args.epochs,
            lr           = args.lr,
            device       = device,
            output_dir   = layer_dir,
        )

        best_row = df_history.loc[df_history["auc_roc_macro"].idxmax()]
        resumen_rows.append({
            "layer":              L,
            "best_epoch":         int(best_row["epoch"]),
            "best_auc_roc_macro": float(best_row["auc_roc_macro"]),
            "f1_macro_at_best":   float(best_row["f1_macro"]),
            "min_val_loss":       float(df_history["val_loss"].min()),
            "no_phase2":          args.no_phase2,
            "epochs":             args.epochs,
            "batch_size":         args.batch_size,
            "lr":                 args.lr,
            "max_train_samples":  args.max_train_samples,
            "max_val_samples":    args.max_val_samples,
            "seed":               args.seed,
            "n_train":            len(train_ds),
            "n_val":              len(val_ds),
            "hidden_size":        hidden_size,
            "output_subdir":      f"layer_{L}",
            "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

        logger.info(
            f"  ✔ Capa {L}: mejor AUC-ROC={best_row['auc_roc_macro']:.4f} "
            f"(época {int(best_row['epoch'])})  F1-macro={best_row['f1_macro']:.4f}"
        )

        # Liberar features de esta capa: ya no se necesitan
        del train_feats[L], val_feats[L]
        gc.collect()

    # ── CSV RESUMEN ÚNICO: mejor valor de cada capa ───────────────────────────
    df_resumen = pd.DataFrame(resumen_rows).sort_values(
        "layer", ascending=False
    ).reset_index(drop=True)
    df_resumen.to_csv(ABLATION_SUMMARY_CSV, index=False)

    # ── Tabla comparativa final + identificación de la mejor capa ─────────────
    best_layer_row = df_resumen.loc[df_resumen["best_auc_roc_macro"].idxmax()]
    logger.info(f"\n{'='*65}")
    logger.info("  RESUMEN — Ablación de Capas Intermedias (mejor AUC-ROC por capa)")
    logger.info(f"{'='*65}")
    logger.info(f"  {'capa':>5} | {'AUC-ROC':>8} | {'F1-macro':>8} | {'val_loss':>8} | época")
    logger.info(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+------")
    for _, r in df_resumen.iterrows():
        mark = "  ← MEJOR" if int(r["layer"]) == int(best_layer_row["layer"]) else ""
        logger.info(
            f"  {int(r['layer']):>5} | {r['best_auc_roc_macro']:>8.4f} | "
            f"{r['f1_macro_at_best']:>8.4f} | {r['min_val_loss']:>8.4f} | "
            f"{int(r['best_epoch']):>4}{mark}"
        )
    logger.info(f"{'='*65}")
    logger.info(
        f"  ★ Mejor capa: {int(best_layer_row['layer'])}  "
        f"(AUC-ROC={best_layer_row['best_auc_roc_macro']:.4f}, "
        f"F1-macro={best_layer_row['f1_macro_at_best']:.4f})"
    )
    logger.info(f"  Resumen CSV  → {ABLATION_SUMMARY_CSV}")
    logger.info(f"  Resultados   → {OUTPUT_DIR}/  (una subcarpeta por capa)")
    logger.info(f"{'='*65}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
