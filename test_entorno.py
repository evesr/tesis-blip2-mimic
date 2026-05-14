#!/usr/bin/env python3
"""
test_entorno.py — Prueba de Vida / Dry Run del pipeline de evaluación.

Verifica que las 4 métricas funcionen correctamente SIN usar GPU ni modelos.
Usa strings médicos hardcodeados para una comprobación rápida.

Ejecutar:
    python test_entorno.py

Métricas verificadas:
    1. BLEU-4      (nltk, smoothing method 1)
    2. ROUGE-L     (rouge_score)
    3. F1-RadGraph (radgraph, reward_level='all', model_type='radgraph-xl')
    4. CheXpert    (labeler ligero, 14 etiquetas, valores 1.0/0.0/-1.0/NaN)
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0. Patch de compatibilidad para radgraph (transformers >=4.x deprecó encode_plus)
# ──────────────────────────────────────────────────────────────────────────────
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
            return_tensors=return_tensors, **kwargs
        )
    PreTrainedTokenizerBase.encode_plus = _encode_plus_compat
    print("[patch] encode_plus añadido a PreTrainedTokenizerBase")

if not hasattr(PreTrainedTokenizerBase, "build_inputs_with_special_tokens"):
    def _build_inputs(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        return ([self.cls_token_id] + token_ids_0 + [self.sep_token_id]
                + token_ids_1 + [self.sep_token_id])
    PreTrainedTokenizerBase.build_inputs_with_special_tokens = _build_inputs
    print("[patch] build_inputs_with_special_tokens añadido")

# ──────────────────────────────────────────────────────────────────────────────
# 1. Imports
# ──────────────────────────────────────────────────────────────────────────────
import math
import sys
import numpy as np

# NLTK — BLEU-4
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Rouge
from rouge_score import rouge_scorer

# RadGraph
from radgraph import F1RadGraph

# CheXpert labeler ligero (implementación incluida abajo)
from chexpert_labeler_lite import CheXpertLabelerLite

# ──────────────────────────────────────────────────────────────────────────────
# 2. Strings de prueba (hardcodeados — NO se usa GPU)
# ──────────────────────────────────────────────────────────────────────────────
REFERENCE = "No pleural effusion or pneumothorax."
PREDICTED  = "There is no pleural effusion or pneumothorax."

SEP = "─" * 65

print(f"\n{'═'*65}")
print("  TEST DE ENTORNO — Pipeline de Evaluación BLIP-2")
print(f"{'═'*65}")
print(f"  Referencia : {REFERENCE}")
print(f"  Predicción : {PREDICTED}")
print(f"{'═'*65}\n")

# ──────────────────────────────────────────────────────────────────────────────
# 3. BLEU-4
# ──────────────────────────────────────────────────────────────────────────────
print(f"[1/4] BLEU-4")
print(SEP)

nltk.download("punkt_tab", quiet=True)
nltk.download("punkt", quiet=True)

ref_tokens  = nltk.word_tokenize(REFERENCE.lower())
pred_tokens = nltk.word_tokenize(PREDICTED.lower())

smoother = SmoothingFunction().method1
bleu4 = sentence_bleu(
    [ref_tokens],
    pred_tokens,
    weights=(0.25, 0.25, 0.25, 0.25),
    smoothing_function=smoother
)
print(f"  BLEU-4 = {bleu4:.4f}\n")

# ──────────────────────────────────────────────────────────────────────────────
# 4. ROUGE-L
# ──────────────────────────────────────────────────────────────────────────────
print(f"[2/4] ROUGE-L")
print(SEP)

scorer_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
rouge_scores = scorer_rouge.score(REFERENCE, PREDICTED)
rouge_l_f1   = rouge_scores["rougeL"].fmeasure
print(f"  ROUGE-L (F1) = {rouge_l_f1:.4f}")
print(f"    Precision   = {rouge_scores['rougeL'].precision:.4f}")
print(f"    Recall      = {rouge_scores['rougeL'].recall:.4f}\n")

# ──────────────────────────────────────────────────────────────────────────────
# 5. F1-RadGraph
# ──────────────────────────────────────────────────────────────────────────────
print(f"[3/4] F1-RadGraph  (reward_level='all', model_type='radgraph-xl')")
print(SEP)

f1radgraph = F1RadGraph(reward_level="all", model_type="radgraph-xl")

mean_reward, reward_list, hyp_ann, ref_ann = f1radgraph(
    refs=[REFERENCE],
    hyps=[PREDICTED]
)

if isinstance(mean_reward, (tuple, list)):
    rg_e, rg_er, rg_bar_er = mean_reward
    print(f"  RG_E   (entidades)               = {rg_e:.4f}")
    print(f"  RG_ER  (entidades + relaciones)   = {rg_er:.4f}")
    print(f"  RG_ER̄  (normalizado completo)     = {rg_bar_er:.4f}")
    radgraph_f1 = float(rg_bar_er)   # métrica principal para el pipeline
else:
    radgraph_f1 = float(mean_reward)
    print(f"  F1-RadGraph = {radgraph_f1:.4f}")

print(f"  → F1-RadGraph principal (RG_ER̄) = {radgraph_f1:.4f}\n")

# ──────────────────────────────────────────────────────────────────────────────
# 6. CheXpert Labeler Ligero
# ──────────────────────────────────────────────────────────────────────────────
print(f"[4/4] CheXpert Labeler (14 etiquetas)")
print(SEP)

labeler = CheXpertLabelerLite()

ref_labels  = labeler.label(REFERENCE)
pred_labels = labeler.label(PREDICTED)

print(f"  {'Etiqueta':<35} {'Referencia':>12}  {'Predicción':>12}")
print(f"  {'─'*35} {'─'*12}  {'─'*12}")
for etiqueta in labeler.LABELS:
    rv = ref_labels[etiqueta]
    pv = pred_labels[etiqueta]
    rv_str = str(rv) if not (isinstance(rv, float) and math.isnan(rv)) else "NaN"
    pv_str = str(pv) if not (isinstance(pv, float) and math.isnan(pv)) else "NaN"
    print(f"  {etiqueta:<35} {rv_str:>12}  {pv_str:>12}")

print()

# ──────────────────────────────────────────────────────────────────────────────
# 7. Resumen
# ──────────────────────────────────────────────────────────────────────────────
print(f"{'═'*65}")
print("  RESUMEN")
print(f"{'═'*65}")
print(f"  ✅ BLEU-4      = {bleu4:.4f}")
print(f"  ✅ ROUGE-L     = {rouge_l_f1:.4f}")
print(f"  ✅ F1-RadGraph = {radgraph_f1:.4f}")
print(f"  ✅ CheXpert    = vector de 14 etiquetas generado")
print(f"{'═'*65}")
print("  Entorno listo para evaluate_final.py\n")
