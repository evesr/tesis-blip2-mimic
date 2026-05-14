#!/usr/bin/env python3
"""
test_entorno.py — Prueba de Vida / Dry Run del pipeline de evaluación.

Verifica que las 4 métricas funcionen correctamente SIN usar la GPU principal
ni los modelos de generación. Usa strings médicos hardcodeados.

    BLEU-4      → nltk (smoothing method 1)
    ROUGE-L     → rouge_score
    F1-RadGraph → radgraph (reward_level='all', model_type='radgraph-xl')
    CheXpert    → CheXbert oficial Stanford AIMI (chexbert.pth)
                  valores oficiales: 1.0 / 0.0 / -1.0 / NaN

Ejecutar:
    python test_entorno.py
"""

# ─────────────────────────────
# 0. Patch de compatibilidad (debe ir ANTES de cualquier import de radgraph/nltk)
# ────────────────
from transformers import PreTrainedTokenizerBase

if not hasattr(PreTrainedTokenizerBase, "encode_plus"):
    def _encode_plus_compat(self, text, text_pair=None, add_special_tokens=True,
                            padding=False, truncation=False, max_length=None,
                            stride=0, return_tensors=None, **kwargs):
        return self.__call__(
            text, text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding=padding, truncation=truncation,
            max_length=max_length, stride=stride,
            return_tensors=return_tensors, **kwargs)
    PreTrainedTokenizerBase.encode_plus = _encode_plus_compat
    print("[patch] encode_plus aplicado")

if not hasattr(PreTrainedTokenizerBase, "build_inputs_with_special_tokens"):
    def _build_inputs(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        return ([self.cls_token_id] + token_ids_0 + [self.sep_token_id]
                + token_ids_1 + [self.sep_token_id])
    PreTrainedTokenizerBase.build_inputs_with_special_tokens = _build_inputs
    print("[patch] build_inputs_with_special_tokens aplicado")

# ──────────
# 1. Imports
# ────────────────
import math
import sys

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from radgraph import F1RadGraph

# Wrapper oficial CheXbert (Stanford AIMI, pesos de RRG_scorers/chexbert.pth)
from chexbert_official import CheXbertLabeler, CHEXBERT_LABELS

# ────────────────
# 2. Strings de hardcodeados (prueba especificación del usuario) 
# ──────────────────
REFERENCE = "No pleural effusion or pneumothorax."
PREDICTED  = "There is no pleural effusion or pneumothorax."

SEP = "─" * 65

print(f"\n{'=':=<65}")
print("  TEST DE ENTORNO — Pipeline de Evaluación BLIP-2")
print(f"{'=':=<65}")
print(f"  Referencia : {REFERENCE}")
print(f"  Prediccion : {PREDICTED}")
print(f"{'=':=<65}\n")

# ───────
# 3. BLEU-4
# ──────────────
print("[1/4] BLEU-4  (nltk, smoothing method 1)")
print(SEP)

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

ref_tokens  = nltk.word_tokenize(REFERENCE.lower())
pred_tokens = nltk.word_tokenize(PREDICTED.lower())

bleu4 = sentence_bleu(
    [ref_tokens], pred_tokens,
    weights=(0.25, 0.25, 0.25, 0.25),
    smoothing_function=SmoothingFunction().method1,
)
print(f"  BLEU-4 = {bleu4:.4f}\n")

# ─────
# 4. ROUGE-L
# ───────────
print("[2/4] ROUGE-L  (rouge_score)")
print(SEP)

rscorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
rscores  = rscorer.score(REFERENCE, PREDICTED)
rouge_l  = rscores["rougeL"].fmeasure

print(f"  ROUGE-L F1  = {rouge_l:.4f}")
print(f"    Precision = {rscores['rougeL'].precision:.4f}")
print(f"    Recall    = {rscores['rougeL'].recall:.4f}\n")

# ───────────────────
# 5. F1-RadGraph  (red neuronal oficial)
# ───────────────────────────────────────────────
print("[3/4] F1-RadGraph  (radgraph, reward_level='all', model_type='radgraph-xl')")
print(SEP)

f1rg = F1RadGraph(reward_level="all", model_type="radgraph-xl")
mean_reward, reward_list, _, _ = f1rg(refs=[REFERENCE], hyps=[PREDICTED])

if isinstance(mean_reward, (tuple, list)):
    rg_e, rg_er, rg_bar_er = mean_reward
    print(f"  RG_E   (solo entidades)              = {rg_e:.4f}")
    print(f"  RG_ER  (entidades + relaciones)       = {rg_er:.4f}")
    print(f"  RG_ERbar (normalizado completo)        = {rg_bar_er:.4f}")
    radgraph_f1 = float(rg_bar_er)
else:
    radgraph_f1 = float(mean_reward)
    print(f"  F1-RadGraph = {radgraph_f1:.4f}")

print(f"  -> Metrica principal (RG_ERbar) = {radgraph_f1:.4f}\n")

# ─────────────────────────────────────────────
# 6. CheXpert — CheXbert oficial Stanford AIMI
# ─────────────────
print("[4/4] CheXpert  (CheXbert oficial Stanford AIMI — chexbert.pth)")
print(SEP)
print("  Cargando modelo CheXbert (BERT-base-uncased + 14 linear heads)...")

labeler = CheXbertLabeler(device="cpu")   # CPU suficiente para dry-run

ref_labels  = labeler.label(REFERENCE)
pred_labels = labeler.label(PREDICTED)


def _fmt(v: float) -> str:
    return "NaN" if (isinstance(v, float) and math.isnan(v)) else f"{v:+.1f}"


print(f"\n  {'Etiqueta':<35} {'Referencia':>12}  {'Prediccion':>12}")
print(f"  {'-'*35} {'-'*12}  {'-'*12}")
for lbl in CHEXBERT_LABELS:
    print(f"  {lbl:<35} {_fmt(ref_labels[lbl]):>12}  {_fmt(pred_labels[lbl]):>12}")

print()

# ────────────────────────────────
# 7. Resumen final
# ────────────────────────────────
print(f"{'=':=<65}")
print("  RESUMEN")
print(f"{'=':=<65}")
print(f"  OK BLEU-4      = {bleu4:.4f}")
print(f"  OK ROUGE-L     = {rouge_l:.4f}")
print(f"  OK F1-RadGraph = {radgraph_f1:.4f}")
print(f"  OK CheXpert    = vector 14 etiquetas generado (CheXbert oficial)")
print(f"{'=':=<65}")
print("  Entorno listo para evaluate_final.py\n")
