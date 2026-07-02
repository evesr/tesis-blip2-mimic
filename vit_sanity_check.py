#!/usr/bin/env python3
"""
vit_sanity_check.py — Linear Probing sobre el ViT de Fase 2 (BLIP-2)
=====================================================================
Sanity Check: ¿Aprende realmente el ViT representaciones clínicas útiles?

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
    python vit_sanity_check.py
    python vit_sanity_check.py --epochs 20 --batch-size 64
    python vit_sanity_check.py --no-phase2   # ablación: ViT base sin Fase 2

Autor : Evelyn Silva Rozas
Fecha : Junio 2026
"""

import argparse
import gc
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

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
IMAGES_DIR  = BASE_DIR / "imagenes_10_porciento"

TRAIN_CSV   = BASE_DIR / "train_split_10pct_limpio.csv"
VAL_CSV     = BASE_DIR / "val_split_10pct_limpio.csv"

CHEXPERT_CSV = BASE_DIR / "mimic-cxr-2.0.0-chexpert.csv"

VIT_LORA_DIR = (
    BASE_DIR
    / "phase2_results-10pct"
    / "checkpoints"
    / "best_model_Config_B1"
    / "vit"
)

OUTPUT_DIR = BASE_DIR / "vit_sanity_check_results"

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
    pixel_values → ViT → CLS token (last_hidden_state[:, 0, :]) → Linear → logits

    El ViT de BLIP-2 (Salesforce/blip2-opt-2.7b) es un EVA-CLIP que devuelve
    un BaseModelOutputWithPooling. Usamos last_hidden_state[:, 0, :] (token CLS)
    como representación global. Si pooler_output está disponible, se usa eso.
    """

    def __init__(self, vit: nn.Module, hidden_size: int, n_classes: int = N_CLASSES):
        super().__init__()
        self.vit        = vit
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [B, 3, H, W]
        outputs = self.vit(pixel_values=pixel_values, return_dict=True)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output          # [B, hidden_size]
        else:
            features = outputs.last_hidden_state[:, 0, :]  # CLS token [B, hidden_size]

        # Cast a float32 para la capa lineal (el ViT puede estar en bfloat16)
        features = features.float()
        return self.classifier(features)              # [B, 14]


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
    vit:        nn.Module,
    dataset:    CheXpertLabeledDataset,
    batch_size: int,
    num_workers: int,
    device:     torch.device,
    split_name: str = "",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pasa TODO el dataset una sola vez por el ViT congelado y almacena
    los embeddings en RAM (CPU, float32).

    Coste: una única pasada forward por época cero → O(N) en vez de O(N*epochs).
    """
    loader = DataLoader(
        dataset,
        batch_size  = batch_size * 2,   # sin grad → batch más grande
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )

    all_features = []
    all_labels   = []

    vit.eval()
    for pixel_values, labels in tqdm(loader, desc=f"  Pre-extrayendo features [{split_name}]"):
        pixel_values = pixel_values.to(device)
        outputs = vit(pixel_values=pixel_values, return_dict=True)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            feats = outputs.pooler_output
        else:
            feats = outputs.last_hidden_state[:, 0, :]

        all_features.append(feats.float().cpu())
        all_labels.append(labels.cpu())

    features = torch.cat(all_features, dim=0)   # [N, hidden_size]
    labels   = torch.cat(all_labels,   dim=0)   # [N, 14]
    logger.info(
        f"  Features [{split_name}]: {features.shape}  "
        f"labels: {labels.shape}  dtype={features.dtype}"
    )
    return features, labels


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
    parser.add_argument("--num-workers",type=int,   default=4,    help="DataLoader workers")
    parser.add_argument("--no-phase2",  action="store_true",      help="Usar ViT base sin pesos de Fase 2 (ablación)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\n{'='*65}")
    logger.info("  VIT SANITY CHECK — Linear Probing CheXpert 14 clases")
    logger.info(f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}")
    logger.info(f"  device={device}  no_phase2={args.no_phase2}")
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

    logger.info("  Cargando Train Dataset (10%)...")
    base_train = MimicCXRDataset(csv_path=TRAIN_CSV, **shared_kwargs)
    logger.info("  Cargando Val Dataset   (10%)...")
    base_val   = MimicCXRDataset(csv_path=VAL_CSV,   **shared_kwargs)

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

    # ── PRE-EXTRAER FEATURES (una sola pasada por el ViT) ────────────────────
    # El ViT está 100% congelado → los embeddings son deterministas.
    # Cachearlos elimina el cuello de botella del forward ViT en cada época.
    logger.info("\n  Pre-extrayendo embeddings del ViT (una sola vez)...")
    train_feats, train_lbls = precompute_features(
        vit, train_ds, args.batch_size, args.num_workers, device, split_name="train"
    )
    val_feats, val_lbls = precompute_features(
        vit, val_ds, args.batch_size, args.num_workers, device, split_name="val"
    )

    # ViT ya no es necesario en GPU → liberar VRAM
    del vit
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("  ViT descargado de VRAM tras extracción de features\n")

    # ── DataLoaders sobre features cacheadas (CPU tensors → muy rápido) ───────
    train_loader = DataLoader(
        FeatureDataset(train_feats, train_lbls),
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = 0,    # tensors en RAM: workers innecesarios
        pin_memory  = True,
    )
    val_loader = DataLoader(
        FeatureDataset(val_feats, val_lbls),
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = True,
    )
    logger.info(f"  Batches train: {len(train_loader):,}  |  Batches val: {len(val_loader):,}\n")

    # ── Clasificador (solo la capa lineal, ViT ya no está en el modelo) ───────
    clf_model = nn.Linear(hidden_size, N_CLASSES).to(device)

    n_trainable = sum(p.numel() for p in clf_model.parameters())
    logger.info(
        f"\n  Clasificador lineal: {n_trainable:,} parámetros entrenables\n"
    )

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    df_history = train(
        model        = clf_model,
        train_loader = train_loader,
        val_loader   = val_loader,
        epochs       = args.epochs,
        lr           = args.lr,
        device       = device,
        output_dir   = OUTPUT_DIR,
    )

    # ── Resumen final ─────────────────────────────────────────────────────────
    best_row = df_history.loc[df_history["auc_roc_macro"].idxmax()]
    logger.info(f"\n{'='*65}")
    logger.info("  RESULTADOS FINALES — Linear Probing ViT (CheXpert 14 clases)")
    logger.info(f"{'='*65}")
    logger.info(f"  Mejor época      : {int(best_row['epoch'])}")
    logger.info(f"  Mejor AUC-ROC    : {best_row['auc_roc_macro']:.4f}")
    logger.info(f"  F1-macro en esa época: {best_row['f1_macro']:.4f}")
    logger.info(f"  Val Loss mínima  : {df_history['val_loss'].min():.4f}")
    logger.info(f"\n  Resultados guardados en: {OUTPUT_DIR}")
    logger.info(f"{'='*65}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
