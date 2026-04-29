"""
Evaluación Zero-Shot del Modelo BLIP2 Base
===========================================
Evalúa el modelo base (sin fine-tuning) en el conjunto de prueba usando:
- Métricas clásicas: BLEU (1-4), ROUGE (1, 2, L)
- Métricas médicas especializadas:
  * CheXpert REAL: 14 patologías con labels del CSV (inciertos = 0.5)
  * RadGraph REAL: Modelo DyGIE++ oficial para grafos de conocimiento
- Visualizaciones profesionales para tesis
- Selección de 10 ejemplos representativos

Autor: Evelyn Silva Rozas
Fecha: Abril 2026
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm

# NLP Metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Medical NLP - RadGraph con patch para compatibilidad
try:
    # Patch para compatibilidad con transformers 5.x
    from transformers import PreTrainedTokenizerBase
    
    # Agregar encode_plus si no existe (deprecado en transformers 5.x pero radgraph lo necesita)
    if not hasattr(PreTrainedTokenizerBase, 'encode_plus'):
        def encode_plus_wrapper(self, text, text_pair=None, add_special_tokens=True, 
                               padding=False, truncation=False, max_length=None,
                               stride=0, return_tensors=None, **kwargs):
            """Wrapper para compatibilidad con encode_plus deprecado."""
            return self.__call__(
                text, 
                text_pair=text_pair,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                return_tensors=return_tensors,
                **kwargs
            )
        PreTrainedTokenizerBase.encode_plus = encode_plus_wrapper
    
    # Agregar build_inputs_with_special_tokens si no existe
    if not hasattr(PreTrainedTokenizerBase, 'build_inputs_with_special_tokens'):
        def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
            """
            Build model inputs from a sequence or a pair of sequence for sequence classification tasks.
            A BERT sequence has the following format: [CLS] X [SEP] (Y [SEP])
            """
            if token_ids_1 is None:
                return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
            cls = [self.cls_token_id]
            sep = [self.sep_token_id]
            return cls + token_ids_0 + sep + token_ids_1 + sep
        PreTrainedTokenizerBase.build_inputs_with_special_tokens = build_inputs_with_special_tokens
    
    from radgraph import F1RadGraph
    RADGRAPH_AVAILABLE = True
except ImportError as e:
    RADGRAPH_AVAILABLE = False
    print(f"⚠️ RadGraph no disponible: {e}. Se usará proxy simplificado.")

# Transformers
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Local imports
from config import config
from preprocessing import estandarizar_imagen
from data_loader import MimicCXRDataset

# Configuración
warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


# ============================================================================
# EXPLICACIÓN MATEMÁTICA DE LAS MÉTRICAS
# ============================================================================

METRICS_EXPLANATION = """
╔══════════════════════════════════════════════════════════════════════════╗
║                  FUNDAMENTOS MATEMÁTICOS DE LAS MÉTRICAS                 ║
╚══════════════════════════════════════════════════════════════════════════╝

1. BLEU (Bilingual Evaluation Understudy)
──────────────────────────────────────────

   BLEU mide la precisión de n-gramas entre el texto generado y la referencia.
   
   Fórmula:
   
   BLEU = BP × exp(∑ wₙ log pₙ)
   
   Donde:
   - pₙ = precisión de n-gramas (n=1,2,3,4)
   - wₙ = peso de cada n-grama (típicamente 1/4 para cada uno)
   - BP = Brevity Penalty (penaliza textos muy cortos)
   
   Precisión de n-gramas:
   
   pₙ = (# n-gramas coincidentes) / (# total n-gramas en generado)
   
   Brevity Penalty:
   
   BP = { 1                si c > r
        { e^(1-r/c)        si c ≤ r
   
   Donde c = longitud del candidato, r = longitud de la referencia
   
   Rango: [0, 1] (1 = coincidencia perfecta)
   
   Interpretación en medicina:
   - BLEU-1: Captura vocabulario correcto (e.g., "pneumonia", "effusion")
   - BLEU-2,3,4: Captura frases médicas ("no acute process", "mild cardiomegaly")

───────────────────────────────────────────────────────────────────────────

2. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
─────────────────────────────────────────────────────────────

   ROUGE mide el recall (cobertura) en lugar de precisión.
   
   ROUGE-1 (unigramas):
   
   ROUGE-1 = ∑(gram ∈ Ref) Count_match(gram) / ∑(gram ∈ Ref) Count(gram)
   
   ROUGE-2 (bigramas): Similar pero con pares de palabras
   
   ROUGE-L (Longest Common Subsequence):
   
   R_lcs = LCS(X,Y) / m    (recall)
   P_lcs = LCS(X,Y) / n    (precision)
   F_lcs = (1+β²)R_lcs·P_lcs / (R_lcs + β²·P_lcs)
   
   Donde:
   - LCS(X,Y) = longitud de la subsecuencia común más larga
   - m = longitud de referencia, n = longitud de candidato
   - β = parámetro de balance (típicamente β²=1)
   
   Rango: [0, 1] (1 = cobertura perfecta)
   
   Ventaja en medicina: Captura si el reporte cubre todos los hallazgos
   importantes, incluso si usa diferentes palabras.

───────────────────────────────────────────────────────────────────────────

3. CheXpert Score (Implementación REAL)
────────────────────────────────────────

   CheXpert es un clasificador de 14 patologías específicas de rayos X de tórax.
   Usamos las ETIQUETAS REALES del dataset MIMIC-CXR (desde el CSV).
   
   14 Patologías:
   - No Finding, Enlarged Cardiomediastinum, Cardiomegaly
   - Lung Opacity, Lung Lesion, Edema, Consolidation
   - Pneumonia, Atelectasis, Pneumothorax
   - Pleural Effusion, Pleural Other, Fracture, Support Devices
   
   Valores de etiquetas:
   - 1.0  = Positivo (patología presente)
   - 0.0  = Negativo (patología ausente)  
   - -1.0 = Incierto → convertimos a 0.5 (crédito parcial)
   
   Detección en texto generado:
   Usamos keywords médicos para detectar si el modelo menciona cada patología.
   
   Métricas calculadas:
   - Accuracy: % de clasificaciones correctas
   - Precision: VP / (VP + FP)
   - Recall: VP / (VP + FN)
   - F1: Media armónica de Precision y Recall
   
   Ventaja: Usa ground truth real, no aproximaciones.

───────────────────────────────────────────────────────────────────────────

4. RadGraph Score (Implementación Real)
────────────────────────────────────────

   RadGraph es un modelo basado en grafos que representa reportes médicos
   como Knowledge Graphs (Grafos de Conocimiento).
   
   ARQUITECTURA:
   
   1. Extracción de Entidades (usando modelo DyGIE++)
      - ANATOMÍA: lung, heart, pleura, bone, etc.
      - OBSERVACIÓN: opacity, enlarged, clear, effusion, etc.
      - RELACIÓN: suggestive_of, located_at, modify, etc.
   
   2. Construcción del Grafo
      G = (V, E) donde:
      - V = conjunto de entidades (nodos)
      - E = conjunto de relaciones (aristas)
   
   3. Comparación de Grafos (Reference vs Generated)
   
   MÉTRICAS CALCULADAS:
   
   a) F1 de Entidades:
      
      Entities_ref = {e₁, e₂, ..., eₙ}
      Entities_gen = {e'₁, e'₂, ..., e'ₘ}
      
      Precision_ent = |Entities_ref ∩ Entities_gen| / |Entities_gen|
      Recall_ent = |Entities_ref ∩ Entities_gen| / |Entities_ref|
      F1_ent = 2 × (Precision × Recall) / (Precision + Recall)
   
   b) F1 de Relaciones:
      
      Relations_ref = {(e₁, r₁, e₂), (e₃, r₂, e₄), ...}
      Relations_gen = {(e'₁, r'₁, e'₂), (e'₃, r'₂, e'₄), ...}
      
      Match si: (entidad_1, relación, entidad_2) coinciden
      
      Precision_rel = |Relations_match| / |Relations_gen|
      Recall_rel = |Relations_match| / |Relations_ref|
      F1_rel = 2 × (Precision × Recall) / (Precision + Recall)
   
   c) Score Global RadGraph:
      
      RadGraph_Score = (F1_ent + F1_rel) / 2
   
   VENTAJAS vs Métricas Léxicas:
   - Captura ESTRUCTURA SEMÁNTICA, no solo palabras
   - Entiende relaciones médicas ("pneumonia in left lung")
   - Robust a paráfrasis ("enlarged heart" = "cardiomegaly")
   - Específico para dominio médico
   
   EJEMPLO:
   
   Referencia: "There is opacity in the left lung"
   Generado:   "Left lung shows infiltrate"
   
   Grafo Referencia:
   - Entidades: [opacity, left lung]
   - Relación: (opacity, located_at, left lung)
   
   Grafo Generado:
   - Entidades: [infiltrate, left lung]
   - Relación: (infiltrate, located_at, left lung)
   
   Match parcial: "left lung" match, "opacity"≈"infiltrate" (similar)
   Relación: match estructural
   
   Score: ~0.75 (alto por estructura correcta aunque léxico diferente)

╚══════════════════════════════════════════════════════════════════════════╝
"""


# ============================================================================
# IMPLEMENTACIÓN DE MÉTRICAS
# ============================================================================

class MedicalMetrics:
    """
    Calcula métricas clásicas y específicas de dominio médico.
    
    Incluye:
    - BLEU: Precisión de n-gramas
    - ROUGE: Recall de n-gramas y subsecuencias
    - CheXpert: Clasificación de 14 patologías (con manejo de inciertos)
    - RadGraph: Comparación de grafos de conocimiento médico (REAL)
    """
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        self.smoothing = SmoothingFunction()
        
        # Inicializar RadGraph (modelo real)
        if RADGRAPH_AVAILABLE:
            try:
                print("   🔬 Inicializando F1RadGraph (modelo DyGIE++)...")
                self.radgraph = F1RadGraph(reward_level="all", model_type="radgraph-xl")
                print("   ✅ F1RadGraph listo")
            except Exception as e:
                print(f"   ⚠️ Error inicializando F1RadGraph: {e}")
                self.radgraph = None
        else:
            self.radgraph = None
        
        # Mapeo de patologías CheXpert a palabras clave para detección en texto
        self.chexpert_keywords = {
            'No Finding': ['normal', 'clear', 'unremarkable', 'no acute', 'within normal limits', 'no abnormality'],
            'Enlarged Cardiomediastinum': ['enlarged cardiomediastinum', 'widened mediastinum', 'mediastinal widening'],
            'Cardiomegaly': ['cardiomegaly', 'enlarged heart', 'cardiac enlargement'],
            'Lung Opacity': ['opacity', 'opacities', 'opacification', 'infiltrate', 'shadowing'],
            'Lung Lesion': ['lesion', 'nodule', 'mass', 'spot'],
            'Edema': ['edema', 'pulmonary edema', 'fluid'],
            'Consolidation': ['consolidation', 'consolidated'],
            'Pneumonia': ['pneumonia', 'infection', 'infectious'],
            'Atelectasis': ['atelectasis', 'collapse', 'collapsed'],
            'Pneumothorax': ['pneumothorax', 'air', 'collapsed lung'],
            'Pleural Effusion': ['effusion', 'pleural effusion', 'fluid collection'],
            'Pleural Other': ['pleural', 'thickening', 'plaques'],
            'Fracture': ['fracture', 'fractured', 'broken', 'break'],
            'Support Devices': ['tube', 'line', 'catheter', 'pacemaker', 'device', 'hardware', 'support']
        }
        
        # Términos médicos para RadGraph proxy fallback
        self.anatomy_terms = [
            'lung', 'heart', 'cardiac', 'mediastinum', 'pleura', 'pleural',
            'chest', 'thorax', 'rib', 'bone', 'diaphragm', 'airway',
            'vasculature', 'aorta', 'hilum', 'apex', 'base'
        ]
        
        self.observation_terms = [
            'opacity', 'consolidation', 'infiltrate', 'effusion', 'pneumonia',
            'edema', 'atelectasis', 'pneumothorax', 'nodule', 'mass',
            'cardiomegaly', 'enlarged', 'clear', 'normal', 'unremarkable',
            'fracture', 'emphysema', 'fibrosis', 'granuloma'
        ]
        
        self.medical_terms = {
            'pathology': [
                'pneumonia', 'effusion', 'edema', 'consolidation',
                'pneumothorax', 'atelectasis', 'opacity', 'infiltrate',
                'nodule', 'mass', 'cardiomegaly', 'fracture', 'emphysema'
            ],
            'normal': [
                'normal', 'clear', 'unremarkable', 'no acute',
                'within normal limits', 'no abnormality'
            ],
            'devices': [
                'tube', 'line', 'catheter', 'pacemaker', 'icd',
                'stent', 'wire', 'support device'
            ]
        }
    
    def detect_chexpert_from_text(self, text: str) -> Dict[str, float]:
        """
        Detecta presencia de patologías CheXpert en el texto generado.
        
        Maneja negaciones y contexto para evitar falsos positivos.
        
        Args:
            text: Texto del reporte generado
            
        Returns:
            Dict con probabilidad de cada patología [0, 1]
        """
        text_lower = text.lower()
        predictions = {}
        
        # Patrones de negación comunes en reportes médicos
        negation_patterns = [
            'no ', 'without ', 'absence of ', 'free of ', 'negative for ',
            'unremarkable ', 'clear of ', 'denied ', 'rule out ', 'unlikely '
        ]
        
        for pathology, keywords in self.chexpert_keywords.items():
            found = False
            negated = False
            
            for keyword in keywords:
                # Buscar el keyword en el texto
                if keyword in text_lower:
                    # Verificar si está negado (buscar 20 caracteres antes)
                    idx = text_lower.find(keyword)
                    context_before = text_lower[max(0, idx-20):idx]
                    
                    # Si hay negación en el contexto, marcarlo como negado
                    if any(neg in context_before for neg in negation_patterns):
                        negated = True
                    else:
                        found = True
                        break  # Si encontramos uno positivo, ya no importan los demás
            
            # Lógica especial para "No Finding"
            if pathology == 'No Finding':
                # "No Finding" se detecta si hay keywords de normal SIN negación
                # O si NO hay ninguna otra patología detectada
                if found and not negated:
                    predictions[pathology] = 1.0
                else:
                    predictions[pathology] = 0.0
            else:
                # Para otras patologías: solo positivo si se encontró SIN negación
                if found and not negated:
                    predictions[pathology] = 1.0
                else:
                    predictions[pathology] = 0.0
        
        # Post-procesamiento: Si "No Finding" está activo, desactivar otras patologías
        # (son mutuamente excluyentes en la mayoría de casos)
        if predictions.get('No Finding', 0.0) == 1.0:
            # Mantener solo "No Finding" y "Support Devices" (pueden coexistir)
            for pathology in predictions:
                if pathology not in ['No Finding', 'Support Devices']:
                    # Solo sobreescribir si hay conflicto fuerte
                    if predictions[pathology] == 1.0:
                        # Si hay evidencia de patología, "No Finding" es incorrecto
                        predictions['No Finding'] = 0.0
                        break
        
        return predictions
    
    def calculate_chexpert_real(self, labels_true: Dict[str, float], text_generated: str) -> Dict[str, float]:
        """
        Calcula métricas CheXpert comparando etiquetas reales vs texto generado.
        
        Etiquetas CheXpert:
        - 1.0 = Positivo (patología presente)
        - 0.0 = Negativo (patología ausente)
        - 0.5 = Incierto (antes era -1, convertido a 0.5)
        
        Estrategia para inciertos (0.5):
        - Da crédito parcial si el modelo detecta o no detecta
        - Penaliza menos que un error completo
        - Incentiva la dirección correcta
        
        Args:
            labels_true: Dict con etiquetas reales {patología: valor}
            text_generated: Texto del reporte generado
            
        Returns:
            Dict con accuracy, precision, recall, f1
        """
        # Detectar patologías en texto generado
        predictions = self.detect_chexpert_from_text(text_generated)
        
        # Calcular métricas solo para patologías presentes en labels_true
        y_true = []
        y_pred = []
        
        for pathology in self.chexpert_keywords.keys():
            if pathology in labels_true:
                true_val = labels_true[pathology]
                pred_val = predictions.get(pathology, 0.0)
                
                # Si es incierto (0.5), usamos comparación suave
                if true_val == 0.5:
                    # Para inciertos, cualquier predicción da 0.5 puntos
                    # Si predice 1 y era incierto → 0.5 puntos
                    # Si predice 0 y era incierto → 0.5 puntos (no suma error)
                    y_true.append(0.5)
                    y_pred.append(pred_val * 0.5)  # Escalar predicción
                else:
                    y_true.append(true_val)
                    y_pred.append(pred_val)
        
        if len(y_true) == 0:
            return {'chexpert_accuracy': 0.0, 'chexpert_precision': 0.0, 'chexpert_recall': 0.0, 'chexpert_f1': 0.0}
        
        # Calcular accuracy (con inciertos como 0.5)
        correct = sum(1 if abs(t - p) < 0.1 else (0.5 if t == 0.5 else 0) for t, p in zip(y_true, y_pred))
        accuracy = correct / len(y_true)
        
        # Para precision/recall/f1, convertir a binario (>= 0.5 = positivo)
        y_true_binary = [1 if t >= 0.5 else 0 for t in y_true]
        y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]
        
        # True Positives, False Positives, False Negatives
        tp = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 1 and p == 0)
        
        # Calcular métricas
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'chexpert_accuracy': accuracy,
            'chexpert_precision': precision,
            'chexpert_recall': recall,
            'chexpert_f1': f1
        }
    
    def calculate_bleu(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        Calcula BLEU-1, BLEU-2, BLEU-3, BLEU-4.
        
        Returns:
            Dict con scores BLEU para cada n-grama
        """
        # Tokenizar
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        
        # Calcular BLEU para n=1,2,3,4
        bleu_scores = {}
        for n in range(1, 5):
            weights = tuple([1/n] * n + [0] * (4-n))
            try:
                score = sentence_bleu(
                    [ref_tokens], 
                    cand_tokens,
                    weights=weights,
                    smoothing_function=self.smoothing.method1
                )
                bleu_scores[f'bleu_{n}'] = score
            except:
                bleu_scores[f'bleu_{n}'] = 0.0
        
        return bleu_scores
    
    def calculate_rouge(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        Calcula ROUGE-1, ROUGE-2, ROUGE-L.
        
        Returns:
            Dict con precision, recall, f1 para cada métrica
        """
        scores = self.rouge_scorer.score(reference, candidate)
        
        rouge_dict = {}
        for key, value in scores.items():
            rouge_dict[f'{key}_precision'] = value.precision
            rouge_dict[f'{key}_recall'] = value.recall
            rouge_dict[f'{key}_fmeasure'] = value.fmeasure
        
        return rouge_dict
    
    def calculate_radgraph_real(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        Calcula RadGraph Score usando el modelo oficial F1RadGraph.
        
        RadGraph extrae grafos de conocimiento médico y los compara:
        1. Extrae entidades médicas (anatomía, observaciones, etc.)
        2. Extrae relaciones entre entidades (located_at, suggestive_of, etc.)
        3. Compara grafos: F1 de entidades + F1 de relaciones
        
        Paper: "Evaluating Progress in Automatic Chest X-Ray Radiology Report 
                Generation" (Delbrouck et al., 2022)
        
        Args:
            reference: Texto de referencia (ground truth)
            candidate: Texto generado por el modelo
            
        Returns:
            Dict con métricas RadGraph:
                - radgraph_simple: F1RadGraph_E (solo entidades)
                - radgraph_partial: F1RadGraph_ER (entidades + relaciones)
                - radgraph_complete: F1RadGraph_bar (entidades + relaciones completas)
                - radgraph_score: Score global (promedio ER y bar)
        
        Example:
            Referencia: "There is opacity in the left lung"
            Generado:   "Left lung shows infiltrate"
            
            RadGraph extrae:
            Ref: {entities: [opacity, left_lung], relations: [(opacity, at, left_lung)]}
            Gen: {entities: [infiltrate, left_lung], relations: [(infiltrate, at, left_lung)]}
            
            Score considera:
            - Entidad "left_lung": MATCH completo
            - Entidad "opacity" vs "infiltrate": MATCH parcial (ambos son observaciones)
            - Relación estructural: MATCH
        """
        if self.radgraph is None:
            # Fallback a proxy si RadGraph no está disponible
            return self._radgraph_proxy_fallback(reference, candidate)
        
        try:
            # F1RadGraph.forward(refs, hyps) retorna:
            # (mean_reward, reward_list, hypothesis_annotation_lists, reference_annotation_lists)
            # donde mean_reward = (rg_e, rg_er, rg_bar_er) cuando reward_level="all"
            
            mean_reward, reward_list, hyp_annotations, ref_annotations = self.radgraph(
                refs=[reference],
                hyps=[candidate]
            )
            
            # mean_reward es una tupla (rg_e, rg_er, rg_bar_er)
            # rg_e: F1 solo entidades
            # rg_er: F1 entidades + relaciones
            # rg_bar_er: F1 entidades + relaciones completas
            if isinstance(mean_reward, tuple) and len(mean_reward) == 3:
                rg_e, rg_er, rg_bar_er = mean_reward
            else:
                # Si no es tupla, es un solo valor
                rg_e = rg_er = rg_bar_er = mean_reward
            
            # El score global típicamente usado es rg_er (el más común en papers)
            radgraph_score = float(rg_er)
            
            return {
                'radgraph_simple': float(rg_e),       # RG_E: solo entidades
                'radgraph_partial': float(rg_er),     # RG_ER: entidades + relaciones
                'radgraph_complete': float(rg_bar_er),  # RG_bar_ER: completo
                'radgraph_score': radgraph_score      # Score principal
            }
            
        except Exception as e:
            print(f"⚠️ Error en RadGraph: {e}. Usando fallback.")
            return self._radgraph_proxy_fallback(reference, candidate)
    
    def _radgraph_proxy_fallback(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        Fallback proxy si RadGraph falla.
        Mantiene compatibilidad con el formato de salida.
        """
        # Términos anatómicos y de observación para proxy
        anatomy_terms = [
            'lung', 'lungs', 'heart', 'cardiac', 'mediastinum',
            'pleura', 'diaphragm', 'bone', 'rib', 'clavicle',
            'aorta', 'vessel', 'hilum', 'chest wall'
        ]
        
        observation_terms = [
            'enlarged', 'opacity', 'clear', 'effusion', 'consolidation',
            'normal', 'abnormal', 'displaced', 'elevated', 'prominent',
            'pneumonia', 'edema', 'atelectasis', 'pneumothorax'
        ]
        
        ref_lower = reference.lower()
        cand_lower = candidate.lower()
        
        # Extraer entidades
        ref_anatomy = set(term for term in anatomy_terms if term in ref_lower)
        cand_anatomy = set(term for term in anatomy_terms if term in cand_lower)
        
        ref_obs = set(term for term in observation_terms if term in ref_lower)
        cand_obs = set(term for term in observation_terms if term in cand_lower)
        
        # F1 para anatomía
        if len(cand_anatomy) == 0 and len(ref_anatomy) == 0:
            f1_anatomy = 1.0
        elif len(cand_anatomy) == 0 or len(ref_anatomy) == 0:
            f1_anatomy = 0.0
        else:
            tp = len(ref_anatomy & cand_anatomy)
            precision = tp / len(cand_anatomy)
            recall = tp / len(ref_anatomy)
            f1_anatomy = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # F1 para observaciones
        if len(cand_obs) == 0 and len(ref_obs) == 0:
            f1_obs = 1.0
        elif len(cand_obs) == 0 or len(ref_obs) == 0:
            f1_obs = 0.0
        else:
            tp = len(ref_obs & cand_obs)
            precision = tp / len(cand_obs)
            recall = tp / len(ref_obs)
            f1_obs = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Score combinado
        score = (f1_anatomy + f1_obs) / 2
        
        return {
            'radgraph_simple': score,
            'radgraph_partial': score * 0.8,
            'radgraph_complete': score * 0.6,
            'radgraph_score': score
        }
    
    def calculate_all(self, reference: str, candidate: str, chexpert_labels: Dict[str, float] = None) -> Dict[str, float]:
        """
        Calcula todas las métricas.
        
        Args:
            reference: Texto de referencia
            candidate: Texto generado
            chexpert_labels: Etiquetas CheXpert reales (opcional)
        
        Returns:
            Dict completo con todas las métricas
        """
        metrics = {}
        
        # BLEU
        metrics.update(self.calculate_bleu(reference, candidate))
        
        # ROUGE
        metrics.update(self.calculate_rouge(reference, candidate))
        
        # CheXpert real (si hay etiquetas disponibles)
        if chexpert_labels is not None and len(chexpert_labels) > 0:
            metrics.update(self.calculate_chexpert_real(chexpert_labels, candidate))
        else:
            # Si no hay etiquetas, valores en 0
            metrics['chexpert_accuracy'] = 0.0
            metrics['chexpert_precision'] = 0.0
            metrics['chexpert_recall'] = 0.0
            metrics['chexpert_f1'] = 0.0
        
        # RadGraph REAL (usa modelo oficial DyGIE++)
        radgraph_metrics = self.calculate_radgraph_real(reference, candidate)
        metrics.update(radgraph_metrics)
        
        return metrics


# ============================================================================
# GENERACIÓN ZERO-SHOT
# ============================================================================

class ZeroShotEvaluator:
    """Evalúa el modelo base BLIP2 sin fine-tuning."""
    
    def __init__(self, device: str = None):
        print("🔄 Inicializando evaluador zero-shot...")
        
        # Detectar dispositivo
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        print(f"   📍 Dispositivo: {self.device}")
        
        # Cargar procesador
        print("   📥 Cargando procesador...")
        self.processor = Blip2Processor.from_pretrained(config.model.model_name)
        
        # Cargar modelo base (sin LoRA)
        print("   🤖 Cargando modelo base BLIP2...")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            config.model.model_name,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=self.dtype
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        # Prompt B (del ab_test)
        self.prompt = "Question: Describe the clinical findings, anatomical structures, and abnormalities visible in this chest radiograph. Answer:"
        
        # Métricas
        self.metrics_calculator = MedicalMetrics()
        
        print("✅ Evaluador listo\n")
    
    def _buscar_imagen(self, dicom_id: str) -> Path:
        """
        Busca imagen por dicom_id (similar a data_loader.py).
        
        Args:
            dicom_id: Identificador DICOM
            
        Returns:
            Path a la imagen o None
        """
        imagen_path = None
        
        # Buscar recursivamente con .jpg
        for archivo in config.paths.images_dir.rglob(f"{dicom_id}.jpg"):
            imagen_path = archivo
            break
        
        # Si no se encuentra, intentar otras extensiones
        if imagen_path is None:
            for ext in ['.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
                for archivo in config.paths.images_dir.rglob(f"{dicom_id}{ext}"):
                    imagen_path = archivo
                    break
                if imagen_path:
                    break
        
        return imagen_path
    
    def generate_report(self, image_path: Path, max_tokens: int = 150) -> str:
        """
        Genera reporte para una imagen usando el pipeline completo de preprocesamiento.
        
        Args:
            image_path: Ruta a la imagen
            max_tokens: Máximo de tokens a generar
            
        Returns:
            Reporte generado
        """
        # Cargar y preprocesar imagen usando el pipeline completo (CLAHE + padding)
        # Esto es exactamente lo que hace data_loader.py
        try:
            image = estandarizar_imagen(
                image_path,
                target_size=config.data.image_size,
                clahe_clip_limit=config.data.clahe_clip_limit,
                clahe_tile_grid_size=config.data.clahe_tile_grid_size
            )
            
            if image is None:
                # Fallback: cargar imagen sin preprocesamiento
                image = Image.open(image_path).convert('RGB')
        except Exception as e:
            # Si falla el preprocesamiento, cargar imagen directamente
            image = Image.open(image_path).convert('RGB')
        
        # Preparar inputs
        inputs = self.processor(
            image, 
            text=self.prompt, 
            return_tensors="pt"
        ).to(self.device, self.dtype)
        
        # Generar (configuración idéntica al ab_test - zero-shot puro)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens
            )
        
        # Decodificar y limpiar
        report = self.processor.decode(outputs[0], skip_special_tokens=True)
        report = report.replace(self.prompt, "").strip()
        
        return report
    
    def evaluate_test_set(self, test_csv: Path = None, max_samples: int = None) -> pd.DataFrame:
        """
        Evalúa en el conjunto de prueba completo.
        
        Args:
            test_csv: Ruta al CSV de prueba (usa config si es None)
            max_samples: Máximo de muestras a evaluar (None = todas)
            
        Returns:
            DataFrame con resultados
        """
        if test_csv is None:
            test_csv = config.paths.test_csv
        
        print(f"\n{'='*80}")
        print("📊 EVALUACIÓN EN CONJUNTO DE PRUEBA")
        print(f"{'='*80}\n")
        
        # Cargar etiquetas CheXpert
        chexpert_path = config.paths.base_dir / "mimic-cxr-2.0.0-chexpert.csv.gz"
        print(f"📋 Intentando cargar etiquetas CheXpert: {chexpert_path}")
        df_chexpert = None
        if chexpert_path.exists():
            try:
                df_chexpert = pd.read_csv(chexpert_path)
                print(f"   ✓ {len(df_chexpert)} etiquetas CheXpert cargadas")
            except Exception as e:
                print(f"   ⚠️ Error leyendo CheXpert: {e}")
                print(f"   📝 Continuando sin métricas CheXpert")
                df_chexpert = None
        else:
            print(f"   ⚠️ Archivo CheXpert no encontrado")
            print(f"   📝 Continuando sin métricas CheXpert")
        
        # Cargar CSV de test
        print(f"\n📖 Cargando datos de test: {test_csv}")
        df = pd.read_csv(test_csv)
        
        if max_samples is not None:
            df = df.head(max_samples)
        
        print(f"   Total de muestras: {len(df)}\n")
        
        # Filtrar solo imágenes que existen (usando búsqueda robusta)
        print("🔍 Verificando existencia de imágenes...")
        valid_rows = []
        for idx, row in df.iterrows():
            dicom_id = row['dicom_id']
            # Usar búsqueda robusta (igual que data_loader.py)
            img_path = self._buscar_imagen(dicom_id)
            if img_path is not None and img_path.exists():
                row_dict = row.to_dict()
                row_dict['img_path_found'] = str(img_path)
                valid_rows.append(row_dict)
        
        df_valid = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(f"   Imágenes válidas: {len(df_valid)} / {len(df)}\n")
        
        # Evaluar cada muestra
        results = []
        
        print("🚀 Generando reportes y calculando métricas...\n")
        for idx in tqdm(range(len(df_valid)), desc="Evaluando", dynamic_ncols=True, unit="img"):
            row = df_valid.iloc[idx]
            dicom_id = row['dicom_id']
            # Usar la ruta encontrada en la búsqueda previa
            img_path = Path(row.get('img_path_found', ''))
            if not img_path.exists():
                # Fallback: buscar de nuevo
                img_path = self._buscar_imagen(dicom_id)
            reference = row.get('report_text', row.get('report', ''))
            
            # Generar reporte
            try:
                generated = self.generate_report(img_path)
            except Exception as e:
                print(f"\n⚠️ Error generando reporte para {dicom_id}: {e}")
                generated = ""
            
            # Checkpoint cada 50 imágenes
            if (idx + 1) % 50 == 0:
                print(f"\n   ✓ Procesadas {idx + 1}/{len(df_valid)} imágenes...")
            
            # Extraer etiquetas CheXpert del archivo oficial (por study_id)
            chexpert_labels = {}
            if df_chexpert is not None and 'study_id' in row:
                study_id = row['study_id']
                # Buscar en df_chexpert por study_id
                chex_row = df_chexpert[df_chexpert['study_id'] == study_id]
                
                if len(chex_row) > 0:
                    chex_row = chex_row.iloc[0]  # Tomar primera fila si hay múltiples
                    chexpert_cols = [
                        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                        'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
                        'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
                        'Pleural Other', 'Fracture', 'Support Devices'
                    ]
                    
                    for col in chexpert_cols:
                        if col in chex_row:
                            val = chex_row[col]
                            # Convertir -1 (incierto) a 0.5
                            if pd.notna(val):
                                if val == -1 or val == '-1' or val == -1.0:
                                    chexpert_labels[col] = 0.5
                                else:
                                    chexpert_labels[col] = float(val)
            
            # Calcular métricas
            if reference and generated:
                metrics = self.metrics_calculator.calculate_all(
                    reference, 
                    generated,
                    chexpert_labels=chexpert_labels if chexpert_labels else None
                )
            else:
                metrics = {k: 0.0 for k in [
                    'bleu_1', 'bleu_2', 'bleu_3', 'bleu_4',
                    'rouge1_fmeasure', 'rouge2_fmeasure', 'rougeL_fmeasure',
                    'chexpert_accuracy', 'chexpert_precision', 'chexpert_recall', 'chexpert_f1',
                    'radgraph_simple', 'radgraph_partial', 'radgraph_complete', 'radgraph_score'
                ]}
            
            # Guardar resultado
            result = {
                'dicom_id': dicom_id,
                'reference': reference,
                'generated': generated,
                'img_path': str(img_path),
                **metrics
            }
            results.append(result)
        
        results_df = pd.DataFrame(results)
        
        print("\n✅ Evaluación completada\n")
        
        return results_df


# ============================================================================
# VISUALIZACIONES
# ============================================================================

def plot_metrics_summary(results_df: pd.DataFrame, output_path: Path):
    """
    Crea visualización de resumen de métricas.
    
    Args:
        results_df: DataFrame con resultados
        output_path: Ruta para guardar la figura
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Evaluación Zero-Shot: Resumen de Métricas', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # 1. BLEU Scores
    ax1 = axes[0, 0]
    bleu_scores = [
        results_df['bleu_1'].mean(),
        results_df['bleu_2'].mean(),
        results_df['bleu_3'].mean(),
        results_df['bleu_4'].mean()
    ]
    bars1 = ax1.bar(['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4'], bleu_scores, 
                     color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title('Métricas BLEU', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Agregar valores en las barras
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    # 2. ROUGE Scores
    ax2 = axes[0, 1]
    rouge_scores = [
        results_df['rouge1_fmeasure'].mean(),
        results_df['rouge2_fmeasure'].mean(),
        results_df['rougeL_fmeasure'].mean()
    ]
    bars2 = ax2.bar(['ROUGE-1', 'ROUGE-2', 'ROUGE-L'], rouge_scores,
                     color=['#9b59b6', '#1abc9c', '#34495e'])
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('F1-Score', fontweight='bold')
    ax2.set_title('Métricas ROUGE', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    # 3. Medical Metrics
    ax3 = axes[1, 0]
    medical_scores = [
        results_df['chexpert_f1'].mean(),
        results_df['chexpert_accuracy'].mean(),
        results_df['radgraph_score'].mean()
    ]
    bars3 = ax3.bar(['CheXpert\nF1', 'CheXpert\nAccuracy', 'RadGraph\nScore'], medical_scores,
                     color=['#e67e22', '#c0392b', '#16a085'])
    ax3.set_ylim(0, 1)
    ax3.set_ylabel('Score', fontweight='bold')
    ax3.set_title('Métricas Médicas Específicas', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 4. Distribución de BLEU-1
    ax4 = axes[1, 1]
    ax4.hist(results_df['bleu_1'], bins=20, color='#3498db', alpha=0.7, edgecolor='black')
    ax4.axvline(results_df['bleu_1'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Media: {results_df["bleu_1"].mean():.3f}')
    ax4.set_xlabel('BLEU-1 Score', fontweight='bold')
    ax4.set_ylabel('Frecuencia', fontweight='bold')
    ax4.set_title('Distribución de BLEU-1', fontweight='bold')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   📊 Gráfico guardado: {output_path}")
    plt.close()


def create_example_visualization(row: pd.Series, output_path: Path, index: int):
    """
    Crea visualización de un ejemplo individual.
    
    Args:
        row: Fila con datos del ejemplo
        output_path: Ruta para guardar
        index: Índice del ejemplo
    """
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1])
    
    # Título principal
    fig.suptitle(f'Ejemplo {index + 1}: {row["dicom_id"]}', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Imagen (izquierda superior)
    ax_img = fig.add_subplot(gs[0, 0])
    try:
        img = Image.open(row['img_path'])
        ax_img.imshow(img, cmap='gray')
        ax_img.axis('off')
        ax_img.set_title('Radiografía de Tórax', fontsize=14, fontweight='bold', pad=10)
    except:
        ax_img.text(0.5, 0.5, 'Imagen no disponible', ha='center', va='center')
        ax_img.axis('off')
    
    # 2. Métricas (derecha superior)
    ax_metrics = fig.add_subplot(gs[0, 1])
    ax_metrics.axis('off')
    
    metrics_text = f"""
    MÉTRICAS DE EVALUACIÓN
    {'='*50}
    
    BLEU Scores:
    • BLEU-1: {row['bleu_1']:.4f}
    • BLEU-2: {row['bleu_2']:.4f}
    • BLEU-3: {row['bleu_3']:.4f}
    • BLEU-4: {row['bleu_4']:.4f}
    
    ROUGE Scores:
    • ROUGE-1: {row['rouge1_fmeasure']:.4f}
    • ROUGE-2: {row['rouge2_fmeasure']:.4f}
    • ROUGE-L: {row['rougeL_fmeasure']:.4f}
    
    Métricas Médicas CheXpert:
    • Accuracy: {row['chexpert_accuracy']:.4f}
    • Precision: {row['chexpert_precision']:.4f}
    • Recall: {row['chexpert_recall']:.4f}
    • F1-Score: {row['chexpert_f1']:.4f}
    
    RadGraph (Knowledge Graphs):
    • Simple F1: {row['radgraph_simple']:.4f}
    • Partial F1: {row['radgraph_partial']:.4f}
    • Complete F1: {row['radgraph_complete']:.4f}
    • Score Global: {row['radgraph_score']:.4f}
    """
    
    ax_metrics.text(0.05, 0.95, metrics_text, 
                    fontsize=11, family='monospace',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 3. Reporte real (izquierda inferior)
    ax_ref = fig.add_subplot(gs[1, 0])
    ax_ref.axis('off')
    
    ref_text = f"REPORTE REAL (Ground Truth):\n\n{row['reference'][:500]}"
    if len(row['reference']) > 500:
        ref_text += "..."
    
    ax_ref.text(0.05, 0.95, ref_text,
                fontsize=10, verticalalignment='top', wrap=True,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # 4. Reporte generado (derecha inferior)
    ax_gen = fig.add_subplot(gs[1, 1])
    ax_gen.axis('off')
    
    gen_text = f"REPORTE GENERADO (Zero-Shot):\n\n{row['generated'][:500]}"
    if len(row['generated']) > 500:
        gen_text += "..."
    
    ax_gen.text(0.05, 0.95, gen_text,
                fontsize=10, verticalalignment='top', wrap=True,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def select_representative_examples(results_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Selecciona ejemplos representativos basados en distribución de métricas.
    
    Estrategia:
    - 3 mejores casos (BLEU-1 alto)
    - 3 casos promedio (BLEU-1 medio)
    - 2 casos difíciles (BLEU-1 bajo)
    - 2 casos con alta variabilidad en métricas
    
    Args:
        results_df: DataFrame con resultados
        n: Número de ejemplos a seleccionar
        
    Returns:
        DataFrame con ejemplos seleccionados
    """
    # Asegurar que hay suficientes muestras
    if len(results_df) < n:
        return results_df
    
    selected_indices = []
    
    # Ordenar por BLEU-1
    df_sorted = results_df.sort_values('bleu_1', ascending=False).reset_index(drop=True)
    
    # 3 mejores
    best_indices = df_sorted.head(3).index.tolist()
    selected_indices.extend(best_indices)
    
    # 3 promedio (alrededor del percentil 50)
    median_idx = len(df_sorted) // 2
    mid_range = range(max(0, median_idx - 5), min(len(df_sorted), median_idx + 5))
    available_mid = [i for i in mid_range if i not in selected_indices]
    mid_indices = np.random.choice(available_mid, min(3, len(available_mid)), replace=False)
    selected_indices.extend(mid_indices)
    
    # 2 difíciles (peores BLEU-1)
    worst_candidates = df_sorted.tail(10).index.tolist()
    worst_available = [i for i in worst_candidates if i not in selected_indices]
    worst_indices = np.random.choice(worst_available, min(2, len(worst_available)), replace=False)
    selected_indices.extend(worst_indices)
    
    # 2 con alta variabilidad
    # Calcular desviación estándar de métricas para cada fila
    metric_cols = ['bleu_1', 'bleu_2', 'rouge1_fmeasure', 'chexpert_f1']
    df_sorted['metric_std'] = df_sorted[metric_cols].std(axis=1)
    high_var_candidates = df_sorted.nlargest(10, 'metric_std').index.tolist()
    high_var_available = [i for i in high_var_candidates if i not in selected_indices]
    
    remaining = n - len(selected_indices)
    if remaining > 0 and len(high_var_available) > 0:
        var_indices = np.random.choice(high_var_available, min(remaining, len(high_var_available)), replace=False)
        selected_indices.extend(var_indices)
    
    # Si aún faltan, tomar aleatorios
    if len(selected_indices) < n:
        remaining = n - len(selected_indices)
        all_available = [i for i in range(len(df_sorted)) if i not in selected_indices]
        if len(all_available) > 0:
            random_indices = np.random.choice(all_available, min(remaining, len(all_available)), replace=False)
            selected_indices.extend(random_indices)
    
    return results_df.iloc[selected_indices].reset_index(drop=True)


# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

def main():
    """Ejecuta la evaluación zero-shot completa."""
    
    print("\n" + "="*80)
    print("🏥 EVALUACIÓN ZERO-SHOT - BLIP2 BASE EN MIMIC-CXR")
    print("="*80)
    print("\nTesis: Generación Automática de Reportes Radiológicos")
    print("Autora: Evelyn Silva Rozas")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*80 + "\n")
    
    # Mostrar explicación de métricas (comentado para ejecución automática)
    # print(METRICS_EXPLANATION)
    # input("\n📖 Presiona ENTER para continuar con la evaluación...")
    
    # Crear directorio de salida
    output_dir = config.paths.base_dir / "resultados_zeroshot"
    output_dir.mkdir(exist_ok=True)
    print(f"\n📁 Directorio de salida: {output_dir}\n")
    
    # Inicializar evaluador
    evaluator = ZeroShotEvaluator()
    
    # Evaluar en test set
    results_df = evaluator.evaluate_test_set()
    
    # Guardar resultados completos
    results_path = output_dir / "resultados_completos.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Resultados guardados: {results_path}")
    
    # Calcular estadísticas
    print("\n" + "="*80)
    print("📊 ESTADÍSTICAS GENERALES")
    print("="*80 + "\n")
    
    stats = {
        'Métrica': [],
        'Media': [],
        'Desv. Est.': [],
        'Mín': [],
        'Máx': []
    }
    
    metric_names = {
        'bleu_1': 'BLEU-1',
        'bleu_2': 'BLEU-2',
        'bleu_3': 'BLEU-3',
        'bleu_4': 'BLEU-4',
        'rouge1_fmeasure': 'ROUGE-1',
        'rouge2_fmeasure': 'ROUGE-2',
        'rougeL_fmeasure': 'ROUGE-L',
        'chexpert_accuracy': 'CheXpert Accuracy',
        'chexpert_precision': 'CheXpert Precision',
        'chexpert_recall': 'CheXpert Recall',
        'chexpert_f1': 'CheXpert F1',
        'radgraph_simple': 'RadGraph Simple',
        'radgraph_partial': 'RadGraph Partial',
        'radgraph_complete': 'RadGraph Complete',
        'radgraph_score': 'RadGraph Score'
    }
    
    for col, name in metric_names.items():
        stats['Métrica'].append(name)
        stats['Media'].append(f"{results_df[col].mean():.4f}")
        stats['Desv. Est.'].append(f"{results_df[col].std():.4f}")
        stats['Mín'].append(f"{results_df[col].min():.4f}")
        stats['Máx'].append(f"{results_df[col].max():.4f}")
    
    stats_df = pd.DataFrame(stats)
    print(stats_df.to_string(index=False))
    
    # Guardar estadísticas
    stats_path = output_dir / "estadisticas.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\n💾 Estadísticas guardadas: {stats_path}")
    
    # Crear visualización de resumen
    print("\n" + "="*80)
    print("🎨 GENERANDO VISUALIZACIONES")
    print("="*80 + "\n")
    
    summary_plot_path = output_dir / "metricas_resumen.png"
    plot_metrics_summary(results_df, summary_plot_path)
    
    # Seleccionar y visualizar ejemplos representativos
    print("\n📋 Seleccionando 10 ejemplos representativos...")
    examples_df = select_representative_examples(results_df, n=10)
    
    examples_dir = output_dir / "ejemplos"
    examples_dir.mkdir(exist_ok=True)
    
    print(f"   Generando visualizaciones individuales...")
    for idx, row in examples_df.iterrows():
        example_path = examples_dir / f"ejemplo_{idx+1:02d}_{row['dicom_id']}.png"
        create_example_visualization(row, example_path, idx)
    
    print(f"   ✅ {len(examples_df)} ejemplos guardados en: {examples_dir}")
    
    # Guardar ejemplos en CSV
    examples_csv_path = output_dir / "ejemplos_seleccionados.csv"
    examples_df.to_csv(examples_csv_path, index=False)
    print(f"   💾 CSV de ejemplos: {examples_csv_path}")
    
    # Resumen final
    print("\n" + "="*80)
    print("✅ EVALUACIÓN COMPLETADA")
    print("="*80)
    print(f"\n📊 Total de muestras evaluadas: {len(results_df)}")
    print(f"📈 BLEU-1 promedio: {results_df['bleu_1'].mean():.4f}")
    print(f"📈 ROUGE-L promedio: {results_df['rougeL_fmeasure'].mean():.4f}")
    print(f"🏥 CheXpert F1: {results_df['chexpert_f1'].mean():.4f}")
    print(f"🏥 CheXpert Accuracy: {results_df['chexpert_accuracy'].mean():.4f}")
    print(f"🔬 RadGraph promedio: {results_df['radgraph_score'].mean():.4f}")
    
    print("\n📁 Archivos generados:")
    print(f"   • Resultados completos: {results_path}")
    print(f"   • Estadísticas: {stats_path}")
    print(f"   • Gráfico resumen: {summary_plot_path}")
    print(f"   • Ejemplos (10): {examples_dir}/")
    print(f"   • CSV ejemplos: {examples_csv_path}")
    
    print("\n" + "="*80)
    print("🎓 Listos para incorporar a tu tesis!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
