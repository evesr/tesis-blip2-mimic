#!/usr/bin/env python3
"""
Constructor del notebook EDA del 50% (eda_50pct_v2.ipynb).

Replica EXACTAMENTE el EDA del 10% (eda_10pct_v2.ipynb) pero:
  • Datos del 50% (train_split_50pct.csv → report_text LIMPIO).
  • Crudo obtenido uniendo por dicom_id con mimic_database_unificada.csv.
  • Rutas de runpod (/workspace/...) y CheXpert en .csv (no .gz).
  • Paleta propia del subconjunto 50% (familia azul) + 100% CPU (sin GPU).
  • Exporta a graficos_eda_exportados/50pct/ y corrige la lista de la Sección 10.

Se usa nbformat para garantizar un .ipynb válido sin escapado manual de JSON.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

# ── 1. Título ────────────────────────────────────────────────────────────────
md("""# 🫁 EDA — Subconjunto MIMIC-CXR 50%
---
**Autora:** Evelyn Silva Rozas · **Fecha:** Junio 2026  
**Descripción:** Análisis Exploratorio de Datos sobre el subconjunto del **50%** del dataset MIMIC-CXR.  
El análisis cubre tres dimensiones:

1. **Estadísticas generales** — tamaño, distribución por estudio y posición de vista.  
2. **Análisis clínico** — etiquetas CheXpert: sanos vs. patología y top 10 enfermedades.  
3. **Análisis NLP** — distribución de tokens y n-gramas (reportes crudos vs. sin paja administrativa).  

> ⚠️ El EDA se realiza **exclusivamente sobre el conjunto Train** para no contaminar val/test.  
> 🖥️ Ejecución **100% en CPU** (no usa GPU) para correr en paralelo al entrenamiento sin OOM.  
> 📁 Todos los gráficos se exportan a `graficos_eda_exportados/50pct/` a 300 DPI.""")

# ── 2. Sección 1 (md) ────────────────────────────────────────────────────────
md("## 📦 Sección 1 — Librerías, Paleta y Configuración de Exportación")

# ── 3. Imports + paleta + CPU-only ───────────────────────────────────────────
code('''import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # EDA 100% en CPU: no toca la GPU (evita OOM y corre en paralelo al entrenamiento)

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore")

# ── Paleta azul — exclusiva para el subconjunto del 50% ─────────────────────
PAL = ["#dbe7f8", "#a8c5ec", "#6f9bdc", "#3a6fc4", "#1b4a9c"]
# Versión extendida interpolada para cuando se necesiten más colores
PAL_EXT = PAL + ["#143a7a", "#0e2b5c", "#081d3e"]

# Colores semánticos fijos (siempre el mismo significado)
C_MAIN    = PAL[2]   # #6f9bdc — barras principales
C_DARK    = PAL[4]   # #1b4a9c — acentos / resaltados
C_LIGHT   = PAL[0]   # #dbe7f8 — fondos / elementos secundarios
C_SANO    = "#4CAF50"
C_PATOL   = PAL[3]   # #3a6fc4

# ── Estilo global ────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi":   150,
    "savefig.dpi":  300,
    "font.family":  "sans-serif",
    "axes.titlepad": 12,
    "axes.labelpad":  7,
})

# ── Carpeta de exportación ───────────────────────────────────────────────────
EXPORT_DIR = "graficos_eda_exportados/50pct"
os.makedirs(EXPORT_DIR, exist_ok=True)

def save_fig(filename: str) -> None:
    """Guarda la figura activa a 300 DPI (PNG) y la muestra en el notebook."""
    path = os.path.join(EXPORT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  ✅ Guardado → {path}")
    plt.show()

print("✓ Configuración lista para subconjunto 50%")
print(f"✓ Exportación → '{EXPORT_DIR}/'")''')

# ── 4. Sección 2 (md) ────────────────────────────────────────────────────────
md("""## 📂 Sección 2 — Carga de Datos

Se carga el split de **Train** del subconjunto 50% (cuyo `report_text` ya está **limpio**, sin paja administrativa)
y se une por `dicom_id` con `mimic_database_unificada.csv` para recuperar el reporte **original (crudo)**.
Las etiquetas CheXpert se unen por `(subject_id, study_id)`.""")

# ── 5. Carga de datos (lógica nueva: unión con base unificada) ───────────────
code('''BASE_DIR = "/workspace/tesis-blip2-mimic"

# ── CheXpert labels ──────────────────────────────────────────────────────────
df_chexpert = pd.read_csv(f"{BASE_DIR}/mimic-cxr-2.0.0-chexpert.csv")
CHEX_COLS   = [c for c in df_chexpert.columns if c not in ("subject_id", "study_id")]

# ── Split Train 50% — su 'report_text' YA está limpio (sin paja administrativa) ──
df_split = pd.read_csv(f"{BASE_DIR}/train_split_50pct.csv")
df_split = df_split.merge(df_chexpert, on=["subject_id", "study_id"], how="left")

# ── Base unificada → reportes ORIGINALES (crudos), unidos por dicom_id ───────
df_unif = pd.read_csv(
    f"{BASE_DIR}/mimic_database_unificada.csv",
    usecols=["dicom_id", "report_text"],
).rename(columns={"report_text": "report_text_crudo"})

# ── Train 50% LIMPIO (split tal cual) ────────────────────────────────────────
df_clean = df_split.copy()
df_clean["tokens"] = df_clean["report_text"].fillna("").apply(lambda x: len(x.split()))

# ── Train 50% CRUDO (mismas filas; report_text reemplazado por el original) ──
df_raw = df_split.merge(df_unif, on="dicom_id", how="left")
n_sin_crudo = int(df_raw["report_text_crudo"].isna().sum())
df_raw["report_text"] = df_raw["report_text_crudo"].fillna(df_raw["report_text"])
df_raw["tokens"] = df_raw["report_text"].fillna("").apply(lambda x: len(x.split()))

print("=" * 55)
print("  DATOS CARGADOS — TRAIN 50%")
print("=" * 55)
print(f"  Imágenes      (crudo):  {len(df_raw):,}")
print(f"  Imágenes      (limpio): {len(df_clean):,}")
print(f"  Estudios únicos:        {df_raw['study_id'].nunique():,}")
print(f"  Pacientes únicos:       {df_raw['subject_id'].nunique():,}")
print(f"  Etiquetas CheXpert:     {len(CHEX_COLS)}")
print(f"  Sin match crudo:        {n_sin_crudo:,}")
print(f"  Tokens promedio (raw):  {df_raw['tokens'].mean():.1f}")
print(f"  Tokens promedio (clean):{df_clean['tokens'].mean():.1f}")
print("=" * 55)''')

# ── 6. Sección 3 (md) ────────────────────────────────────────────────────────
md("## 📊 Sección 3 — Estadísticas Generales del Conjunto Train")

# ── 7. Métricas ──────────────────────────────────────────────────────────────
code('''metricas = {
    "Imágenes totales":                      len(df_raw),
    "Estudios únicos":                       df_raw["study_id"].nunique(),
    "Pacientes únicos":                      df_raw["subject_id"].nunique(),
    "Vistas distintas":                      df_raw["ViewPosition"].nunique(),
    "Tokens medio (original)":               round(df_raw["tokens"].mean(), 1),
    "Tokens mediana (original)":             int(df_raw["tokens"].median()),
    "Tokens P90 (original)":                 int(df_raw["tokens"].quantile(0.90)),
    "Tokens P95 (original)":                 int(df_raw["tokens"].quantile(0.95)),
    "Tokens P99 (original)":                 int(df_raw["tokens"].quantile(0.99)),
    "Tokens medio (preprocesado)":           round(df_clean["tokens"].mean(), 1),
    "Tokens mediana (preprocesado)":         int(df_clean["tokens"].median()),
    "Reducción media tras preprocesamiento": round(df_raw["tokens"].mean() - df_clean["tokens"].mean(), 1),
}
tabla = pd.DataFrame(metricas.items(), columns=["Métrica", "Valor"])
tabla["Valor"] = tabla["Valor"].apply(lambda v: f"{v:,}" if isinstance(v, int) else v)
print(tabla.to_string(index=False))''')

# ── 8. Sección 4 (md) ────────────────────────────────────────────────────────
md("""## 🖼️ Sección 4 — Distribución de Imágenes por Estudio

Un estudio puede tener múltiples imágenes (AP + lateral, etc.). Esta distribución muestra cuántos estudios tienen 1, 2, 3… imágenes.

> 📁 `distribucion_imagenes_por_estudio.png`""")

# ── 9. Imágenes por estudio ──────────────────────────────────────────────────
code('''imgs_por_estudio = df_raw.groupby("study_id")["dicom_id"].count().value_counts().sort_index()

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(
    imgs_por_estudio.index.astype(str),
    imgs_por_estudio.values,
    color=PAL[2], edgecolor=PAL[4], linewidth=0.8, alpha=0.90,
)
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + imgs_por_estudio.max() * 0.012,
            f"{int(h):,}", ha="center", va="bottom", fontsize=9, fontweight="bold", color=PAL[4])

ax.set_title("Distribución de Imágenes por Estudio — Train 50%", fontsize=13, fontweight="bold")
ax.set_xlabel("Número de imágenes por estudio", fontsize=11)
ax.set_ylabel("Número de estudios", fontsize=11)
ax.grid(axis="y", alpha=0.35)
plt.tight_layout()
save_fig("distribucion_imagenes_por_estudio.png")
print(imgs_por_estudio.rename("estudios").to_frame().to_string())''')

# ── 10. Sección 5 (md) ───────────────────────────────────────────────────────
md("""## 📐 Sección 5 — Proporción de Posición de Vista (ViewPosition)

Distribución de las vistas radiológicas: **PA**, **AP**, **LATERAL** y **LL**.

> 📁 `proporcion_vistas.png`""")

# ── 11. ViewPosition ─────────────────────────────────────────────────────────
code('''counts = df_raw["ViewPosition"].fillna("Desconocido").value_counts()
n_vistas = len(counts)
colores  = (PAL_EXT + PAL_EXT)[:n_vistas]
total    = counts.sum()

# ── Tabla resumen ─────────────────────────────────────────────────────────────
tabla_vistas = pd.DataFrame({
    "Vista": counts.index,
    "N° imágenes": counts.values,
    "Porcentaje (%)": (counts.values / total * 100).round(2),
})
print("Distribución de posición de vista — Train 50%")
print(tabla_vistas.to_string(index=False))
print()

# ── Gráfico de barras ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(counts.index, counts.values,
              color=colores, edgecolor=PAL[4], linewidth=0.8, alpha=0.90)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.003,
            f"{val/total*100:.1f}%\\n({val:,})",
            ha="center", va="bottom", fontsize=9)
ax.set_xlabel("Posición de vista", fontsize=11)
ax.set_ylabel("Número de imágenes", fontsize=11)
ax.set_title("Distribución de Posición de Vista — Train 50%", fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.35)
plt.tight_layout()
save_fig("proporcion_vistas.png")''')

# ── 12. Sección 6 (md) ───────────────────────────────────────────────────────
md("""## 🩺 Sección 6 — Sanos vs. Patología (`No Finding`)

`No Finding = 1` → radiografía normal. Los demás casos se consideran con patología o sin etiquetar.

> 📁 `sanos_vs_patologia.png`""")

# ── 13. Sanos vs patología ───────────────────────────────────────────────────
code('''sano      = int((df_raw["No Finding"] == 1).sum())
patologia = len(df_raw) - sano
n_total   = len(df_raw)

# ── Tabla resumen ─────────────────────────────────────────────────────────────
tabla_diagnostico = pd.DataFrame({
    "Categoría":        ["Sin hallazgos patológicos", "Con hallazgos patológicos"],
    "N° imágenes":      [sano, patologia],
    "Porcentaje (%)":   [round(sano/n_total*100, 2), round(patologia/n_total*100, 2)],
})
print("Distribución diagnóstica — Train 50%")
print(tabla_diagnostico.to_string(index=False))
print()

# ── Gráfico de barras ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
categorias = ["Sin hallazgos\\npatológicos", "Con hallazgos\\npatológicos"]
bars = ax.bar(categorias, [sano, patologia],
              color=[C_SANO, C_PATOL], edgecolor=PAL[4], linewidth=0.8, alpha=0.90,
              width=0.5)
for bar, val in zip(bars, [sano, patologia]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + n_total * 0.005,
            f"{val/n_total*100:.1f}%\\n({val:,})",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Número de imágenes", fontsize=11)
ax.set_title("Distribución Diagnóstica — Train 50%", fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.35)
plt.tight_layout()
save_fig("sanos_vs_patologia.png")''')

# ── 14. Sección 7 (md) ───────────────────────────────────────────────────────
md("""## 🏥 Sección 7 — Top 10 Enfermedades más Frecuentes

Prevalencia de cada etiqueta CheXpert (`= 1.0`) como porcentaje del total de imágenes. Se excluye `No Finding`.

> 📁 `top_10_enfermedades.png`""")

# ── 15. Top 10 enfermedades ──────────────────────────────────────────────────
code('''enf_cols = [c for c in CHEX_COLS if c != "No Finding" and c in df_raw.columns]
prevalencia = pd.Series(
    {col: (df_raw[col] == 1.0).sum() / len(df_raw) * 100 for col in enf_cols}
).sort_values(ascending=True).tail(10)

norm      = plt.Normalize(prevalencia.min(), prevalencia.max())
cmap_vals = plt.cm.colors.LinearSegmentedColormap.from_list("pal", PAL)
bar_cols  = [cmap_vals(norm(v)) for v in prevalencia.values]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(len(prevalencia)), prevalencia.values,
               color=bar_cols, edgecolor=PAL[4], linewidth=0.7, alpha=0.92)

ax.set_yticks(range(len(prevalencia)))
ax.set_yticklabels(prevalencia.index, fontsize=10)
ax.set_xlabel("Prevalencia (%)", fontsize=11)
ax.set_title("Top 10 Enfermedades más Frecuentes — Train 50%", fontsize=13, fontweight="bold")
ax.set_xlim(0, prevalencia.max() * 1.22)
ax.grid(axis="x", alpha=0.35)

for bar, val in zip(bars, prevalencia.values):
    ax.text(val + prevalencia.max() * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=9, fontweight="bold", color=PAL[4])

plt.tight_layout()
save_fig("top_10_enfermedades.png")
print("\\nPrevalencia top 10:")
print(prevalencia.sort_values(ascending=False).round(2).to_string())''')

# ── 16. Sección 8 (md) ───────────────────────────────────────────────────────
md("""## 📝 Sección 8 — Distribución de Tokens

Histogramas de la distribución de tokens por reporte para el texto **original** y el texto **preprocesado**.  
La línea vertical naranja indica la **media** y la verde la **mediana**.  
La diferencia entre ambas distribuciones cuantifica el efecto del preprocesamiento sobre la extensión textual.

> 📁 `distribucion_tokens_original.png` · `distribucion_tokens_preprocesado.png`""")

# ── 17. Tokens ───────────────────────────────────────────────────────────────
code('''for df_tok, etiqueta, fname_tok, color in [
    (df_raw,   "Texto original",       "distribucion_tokens_original.png",      PAL[2]),
    (df_clean, "Texto preprocesado",   "distribucion_tokens_preprocesado.png",  PAL[3]),
]:
    t   = df_tok["tokens"]
    mu  = t.mean()
    med = t.median()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(t, bins=50, color=color, edgecolor="white", linewidth=0.4, alpha=0.85)
    ax.axvline(mu,  color="#FF6F00", linewidth=2, linestyle="--", label=f"Media: {mu:.0f} tokens")
    ax.axvline(med, color="#2E7D32", linewidth=2, linestyle="-",  label=f"Mediana: {med:.0f} tokens")
    ax.set_title(f"Distribución de Tokens — {etiqueta} (Train 50%)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Número de tokens por reporte", fontsize=11)
    ax.set_ylabel("Frecuencia", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_fig(fname_tok)

redu = df_raw["tokens"].mean() - df_clean["tokens"].mean()
print(f"\\n  Reducción promedio tras preprocesamiento: {redu:.1f} tokens ({redu/df_raw['tokens'].mean()*100:.1f}%)")''')

# ── 18. Sección 8b (md) ──────────────────────────────────────────────────────
md("""## 📊 Sección 8b — Distribución de Diversidad Léxica (TTR)

La **diversidad léxica** se mide mediante el *Type-Token Ratio* (TTR):

$$\\text{TTR} = \\frac{\\text{palabras únicas}}{\\text{palabras totales}}$$

Un TTR cercano a 1 indica vocabulario muy variado; un valor bajo señala repetición.  
Se compara la distribución del TTR en el texto **original** vs. **preprocesado**.

> 📁 `diversidad_lexica_original.png` · `diversidad_lexica_preprocesado.png`""")

# ── 19. TTR ──────────────────────────────────────────────────────────────────
code('''def calc_ttr(text: str) -> float:
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)

for df_ttr, etiqueta, fname_ttr, color in [
    (df_raw,   "Texto original",     "diversidad_lexica_original.png",     PAL[1]),
    (df_clean, "Texto preprocesado", "diversidad_lexica_preprocesado.png", PAL[3]),
]:
    ttr = df_ttr["report_text"].fillna("").apply(calc_ttr)
    mu_ttr  = ttr.mean()
    med_ttr = ttr.median()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(ttr, bins=50, color=color, edgecolor="white", linewidth=0.4, alpha=0.85)
    ax.axvline(mu_ttr,  color="#FF6F00", linewidth=2, linestyle="--",
               label=f"Media: {mu_ttr:.3f}")
    ax.axvline(med_ttr, color="#2E7D32", linewidth=2, linestyle="-",
               label=f"Mediana: {med_ttr:.3f}")
    ax.set_title(f"Distribución de Diversidad Léxica (TTR) — {etiqueta} (Train 50%)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Type-Token Ratio (TTR)", fontsize=11)
    ax.set_ylabel("Frecuencia", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_fig(fname_ttr)

print(f"\\n  TTR medio — Original   : {df_raw['report_text'].fillna('').apply(calc_ttr).mean():.3f}")
print(f"  TTR medio — Preprocesado: {df_clean['report_text'].fillna('').apply(calc_ttr).mean():.3f}")''')

# ── 20. Sección 9 (md) ───────────────────────────────────────────────────────
md("""## 🔤 Sección 9 — Análisis de N-gramas: Texto Original vs. Preprocesado

Cada tipo de n-grama genera **dos gráficos independientes**:
- **Original** — Top 20 n-gramas del reporte sin modificar (incluye encabezados, indicaciones clínicas, referencias de comparación, etc.)
- **Preprocesado** — Top 20 n-gramas tras la normalización del texto (contenido clínico depurado)

La comparación permite verificar qué términos no clínicos son eliminados y que el vocabulario médico relevante se preserva.

> 📁 `ngramas_unigramas_original.png` · `ngramas_unigramas_preprocesado.png`  
> 📁 `ngramas_bigramas_original.png` · `ngramas_bigramas_preprocesado.png`  
> 📁 `ngramas_trigramas_original.png` · `ngramas_trigramas_preprocesado.png`""")

# ── 21. Funciones n-gramas ───────────────────────────────────────────────────
code('''def get_top_ngrams(corpus: pd.Series, n: int, top_k: int = 20) -> pd.DataFrame:
    """
    Extrae los top_k n-gramas más frecuentes de un corpus de textos.
    """
    vec = CountVectorizer(
        ngram_range=(n, n),
        max_features=5000,
        lowercase=True,
        stop_words=None,
    )
    X    = vec.fit_transform(corpus.fillna(""))
    freq = X.sum(axis=0).A1
    df   = pd.DataFrame({"ngram": vec.get_feature_names_out(), "freq": freq})
    return df.nlargest(top_k, "freq").sort_values("freq")


def plot_ngrams_single(
    corpus: pd.Series,
    n: int,
    titulo_tipo: str,
    etiqueta_corpus: str,
    filename: str,
    color: str,
    subtitulo: str = "Train 50%",
    top_k: int = 20,
) -> None:
    """
    Genera un único gráfico de barras horizontales con los top_k n-gramas
    más frecuentes de un corpus.
    """
    df_ng = get_top_ngrams(corpus, n, top_k)

    fig, ax = plt.subplots(figsize=(11, 8))
    bars = ax.barh(range(len(df_ng)), df_ng["freq"].values,
                   color=color, edgecolor=PAL[4], linewidth=0.6, alpha=0.88)
    ax.set_yticks(range(len(df_ng)))
    ax.set_yticklabels(df_ng["ngram"].values, fontsize=9)
    ax.set_xlabel("Frecuencia", fontsize=11)
    ax.set_title(
        f"Top {top_k} {titulo_tipo} — {etiqueta_corpus}\\n{subtitulo}",
        fontsize=13, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, df_ng["freq"].max() * 1.18)
    for bar, val in zip(bars, df_ng["freq"].values):
        ax.text(val + df_ng["freq"].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{int(val):,}", va="center", fontsize=8, color=PAL[4])
    plt.tight_layout()
    save_fig(filename)


# ── Corpus ──────────────────────────────────────────────────────────────────
corpus_raw   = df_raw["report_text"]
corpus_clean = df_clean["report_text"]

print("✓ Función definida. Ejecuta las celdas siguientes para generar cada tipo de n-grama.")''')

# ── 22. 9a (md) ──────────────────────────────────────────────────────────────
md("### 9a — Unigramas (palabras individuales)")

# ── 23. Unigramas ────────────────────────────────────────────────────────────
code('''plot_ngrams_single(corpus_raw,   n=1, titulo_tipo="Unigramas", etiqueta_corpus="Texto original",     filename="ngramas_unigramas_original.png",     color=PAL[1])
plot_ngrams_single(corpus_clean, n=1, titulo_tipo="Unigramas", etiqueta_corpus="Texto preprocesado", filename="ngramas_unigramas_preprocesado.png", color=PAL[3])''')

# ── 24. 9b (md) ──────────────────────────────────────────────────────────────
md("### 9b — Bigramas (pares de palabras)")

# ── 25. Bigramas ─────────────────────────────────────────────────────────────
code('''plot_ngrams_single(corpus_raw,   n=2, titulo_tipo="Bigramas", etiqueta_corpus="Texto original",     filename="ngramas_bigramas_original.png",     color=PAL[1])
plot_ngrams_single(corpus_clean, n=2, titulo_tipo="Bigramas", etiqueta_corpus="Texto preprocesado", filename="ngramas_bigramas_preprocesado.png", color=PAL[3])''')

# ── 26. 9c (md) ──────────────────────────────────────────────────────────────
md("### 9c — Trigramas (secuencias de tres palabras)")

# ── 27. Trigramas ────────────────────────────────────────────────────────────
code('''plot_ngrams_single(corpus_raw,   n=3, titulo_tipo="Trigramas", etiqueta_corpus="Texto original",     filename="ngramas_trigramas_original.png",     color=PAL[1])
plot_ngrams_single(corpus_clean, n=3, titulo_tipo="Trigramas", etiqueta_corpus="Texto preprocesado", filename="ngramas_trigramas_preprocesado.png", color=PAL[3])''')

# ── 28. Sección 10 (md) ──────────────────────────────────────────────────────
md("## ✅ Sección 10 — Resumen de Archivos Exportados")

# ── 29. Resumen (lista CORREGIDA con los nombres reales) ─────────────────────
code('''archivos_esperados = [
    "distribucion_imagenes_por_estudio.png",
    "proporcion_vistas.png",
    "sanos_vs_patologia.png",
    "top_10_enfermedades.png",
    "distribucion_tokens_original.png",
    "distribucion_tokens_preprocesado.png",
    "diversidad_lexica_original.png",
    "diversidad_lexica_preprocesado.png",
    "ngramas_unigramas_original.png",
    "ngramas_unigramas_preprocesado.png",
    "ngramas_bigramas_original.png",
    "ngramas_bigramas_preprocesado.png",
    "ngramas_trigramas_original.png",
    "ngramas_trigramas_preprocesado.png",
]

print(f"Archivos exportados en '{EXPORT_DIR}/':\\n")
todos_ok = True
for fname in archivos_esperados:
    fpath = os.path.join(EXPORT_DIR, fname)
    existe = os.path.isfile(fpath)
    size   = f"{os.path.getsize(fpath)/1024:.0f} KB" if existe else "—"
    estado = "✅" if existe else "❌ FALTA"
    if not existe:
        todos_ok = False
    print(f"  {estado}  {fname:<45} {size}")

print()
print(f"Total: {sum(os.path.isfile(os.path.join(EXPORT_DIR, f)) for f in archivos_esperados)}/{len(archivos_esperados)} gráficos PNG")
print("✅ EDA 50% completo." if todos_ok else "⚠️  Algunos archivos no se generaron. Revisa las celdas anteriores.")''')

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

out = "/workspace/tesis-blip2-mimic/eda_50pct_v2.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✓ Notebook creado: {out}  ({len(cells)} celdas)")
