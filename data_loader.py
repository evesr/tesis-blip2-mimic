"""
Módulo de carga de datos para el dataset MIMIC-CXR.

Implementa el Dataset de PyTorch para cargar imágenes radiológicas
y sus reportes asociados desde archivos CSV.

Author: [Tu nombre]
Date: 2026-04-22
"""

import os
import time
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, Union
from PIL import Image
import logging

from torchvision import transforms

from preprocessing import estandarizar_imagen, crear_imagen_placeholder
from config import config

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MimicCXRDataset(Dataset):
    """
    Dataset de PyTorch para MIMIC-CXR con preprocesamiento integrado.
    
    Este dataset carga imágenes de radiografías de tórax junto con sus reportes
    médicos asociados desde un archivo CSV. Aplica preprocesamiento automático
    a las imágenes (CLAHE, padding, etc.) y tokeniza los textos usando el
    procesador de BLIP2.
    
    Attributes:
        csv_path: Ruta al archivo CSV con metadata
        images_dir: Directorio raíz donde se encuentran las imágenes
        processor: Procesador de BLIP2 para tokenización de imágenes/texto
        data: DataFrame con los datos cargados del CSV
        image_size: Tamaño objetivo de las imágenes
        
    Example:
        >>> from transformers import Blip2Processor
        >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        >>> dataset = MimicCXRDataset(
        ...     csv_path="train_split_limpio.csv",
        ...     images_dir="imagenes_1_porciento",
        ...     processor=processor
        ... )
        >>> sample = dataset[0]
        >>> print(sample.keys())
    """
    
    def __init__(
        self,
        csv_path: Union[str, Path],
        images_dir: Union[str, Path],
        processor: Any,
        image_size: Tuple[int, int] = (224, 224),
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        max_length: int = 512,
        padding: str = "max_length",
        prompt: str = None,
        augment: bool = False,
        debug_log_samples: int = 3
    ):
        """
        Inicializa el dataset.
        
        Args:
            csv_path: Ruta al archivo CSV con columnas: dicom_id, report_text, etc.
            images_dir: Directorio raíz donde buscar las imágenes
            processor: Procesador BLIP2 para procesar imágenes y texto
            image_size: Tamaño objetivo para las imágenes (ancho, alto)
            clahe_clip_limit: Límite de clip para CLAHE
            clahe_tile_grid_size: Tamaño de grid para CLAHE
            max_length: Longitud máxima de secuencia para texto
            padding: Estrategia de padding para texto
            prompt: Prompt base opcional; si se proporciona, se usa como texto de
                entrada (text) y el reporte como text_target (labels) — modo seq2seq
            augment: Si True, aplica Data Augmentation espacial AL VUELO en
                __getitem__ (RandomRotation ±5° + RandomAffine translate ±5%),
                DESPUÉS de CLAHE+padding y ANTES del Blip2Processor. Debe ser
                True SOLO para el split de ENTRENAMIENTO; False en val/test/
                inferencia para una evaluación determinista y limpia.
            debug_log_samples: Número de muestras iniciales (por worker) para las
                que __getitem__ emite un log detallado de trazabilidad.
            
        Raises:
            FileNotFoundError: Si el CSV o el directorio de imágenes no existe
            ValueError: Si el CSV no contiene las columnas requeridas
        """
        self.csv_path = Path(csv_path)
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.image_size = image_size
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.max_length = max_length
        self.padding = padding
        self.prompt = prompt

        # ── Data Augmentation Espacial (solo ENTRENAMIENTO) ──────────────────
        # Se aplica AL VUELO en __getitem__, DESPUÉS de CLAHE+padding y ANTES del
        # Blip2Processor. SOLO transformaciones GEOMÉTRICAS clínicamente seguras:
        #   • RandomRotation(±5°)            → tolera la ligera inclinación del paciente
        #   • RandomAffine(translate ±5% XY) → tolera el descentrado del tórax
        # PROHIBIDO (decisión clínica explícita):
        #   ✘ ColorJitter/brillo/contraste → destruiría el realce CLAHE ya aplicado
        #   ✘ RandomCrop                    → podría recortar hallazgos periféricos
        # fill=0 (negro) = consistente con el padding negro del X-ray.
        self.augment = augment
        self.debug_log_samples = debug_log_samples
        self._debug_logged = 0  # contador interno; se reinicia en cada worker (fork)

        if self.augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomRotation(
                    degrees=5,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    fill=0,
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    fill=0,
                ),
            ])
        else:
            self.augment_transform = None
        
        # Validar existencia de archivos
        if not self.csv_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo CSV: {self.csv_path}")
        
        if not self.images_dir.exists():
            raise FileNotFoundError(f"No se encontró el directorio de imágenes: {self.images_dir}")
        
        # Cargar datos del CSV
        logger.info(f"📘 Cargando dataset desde: {self.csv_path}")
        self.data = pd.read_csv(self.csv_path)
        
        # Validar columnas requeridas
        required_columns = ['dicom_id', 'report_text']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            raise ValueError(
                f"El CSV no contiene las columnas requeridas: {missing_columns}. "
                f"Columnas disponibles: {list(self.data.columns)}"
            )
        
        # Limpiar datos: eliminar filas sin imagen o sin reporte
        original_size = len(self.data)
        self.data = self.data.dropna(subset=['dicom_id', 'report_text'])
        self.data = self.data[self.data['report_text'].str.strip() != '']
        
        if len(self.data) < original_size:
            logger.warning(
                f"⚠️ Se eliminaron {original_size - len(self.data)} filas "
                f"con datos faltantes"
            )
        
        # Resetear índice
        self.data = self.data.reset_index(drop=True)
        
        logger.info(f"✅ Dataset inicializado con {len(self.data)} muestras")

        # ── OPTIMIZACIÓN I/O: índice {dicom_id: ruta} construido UNA sola vez ──
        # Antes: _buscar_imagen hacía un rglob RECURSIVO sobre TODO el árbol de
        # imágenes (~191K archivos) en CADA __getitem__ → O(N) por muestra,
        # O(N²) por época: el cuello de botella dominante del entrenamiento.
        # Ahora: un único recorrido del árbol en __init__ → lookups O(1) en RAM.
        # Se construye ANTES del fork de los DataLoader workers, así que se
        # hereda por copy-on-write (no se reconstruye por worker).
        self._indice_imagenes = self._construir_indice_imagenes()

        # ── OPTIMIZACIÓN CPU: prompts pre-tokenizados por vista ────────────────
        # El prompt_text solo depende de la vista (3 valores: Frontal/Lateral/
        # Unknown).  Pre-tokenizamos los 3 una vez para no llamar al tokenizer
        # por cada muestra al enmascarar el prompt en los labels.
        self._prompt_text_cache: Dict[str, str] = {}
        self._prompt_ntokens_cache: Dict[str, int] = {}
        if self.prompt is not None:
            base_prompt = self.prompt
            for _vista in ("Frontal", "Lateral", "Unknown"):
                _pt = f"[Context: {_vista} view] {base_prompt}"
                self._prompt_text_cache[_vista] = _pt
                self._prompt_ntokens_cache[_vista] = len(
                    self.processor.tokenizer(_pt, add_special_tokens=False)["input_ids"]
                )

        # ── Log detallado de configuración (máxima trazabilidad) ──────────────
        _ejemplo_prompt = (
            f"[Context: Frontal view] "
            f"{self.prompt if self.prompt is not None else config.inference.default_prompt}"
        )
        logger.info("🧾 ─── Config Dataset (%s) ────────────────────────────", self.csv_path.name)
        logger.info("    • image_size=%s | CLAHE(clip=%.2f, grid=%s) | padding=%s | max_length=%d",
                    self.image_size, self.clahe_clip_limit, self.clahe_tile_grid_size,
                    self.padding, self.max_length)
        logger.info("    • prompt_base=%s",
                    "config.inference.default_prompt" if self.prompt is None
                    else "explícito (arg prompt=)")
        logger.info("    • estructura_prompt='[Context: {vista} view] {prompt_base}'")
        logger.info("    • ejemplo_prompt=%r", _ejemplo_prompt)
        logger.info("    • índice_imágenes=%d rutas | prompts_pretokenizados=%d vistas",
                    len(self._indice_imagenes), len(self._prompt_ntokens_cache))
        if self.augment:
            logger.info("    • 🔄 DATA AUGMENTATION: ACTIVADA (split de ENTRENAMIENTO)")
            logger.info("        - RandomRotation(degrees=±5, interp=BILINEAR, fill=0)")
            logger.info("        - RandomAffine(degrees=0, translate=±5% XY, interp=BILINEAR, fill=0)")
            logger.info("        - SIN ColorJitter (protege CLAHE) · SIN RandomCrop (protege hallazgos)")
            logger.info("        - Orden: imagen → CLAHE+padding → AUGMENT → Blip2Processor")
        else:
            logger.info("    • 🔒 DATA AUGMENTATION: DESACTIVADA (val/test/inferencia → determinista)")
    
    def __len__(self) -> int:
        """
        Retorna el número total de muestras en el dataset.
        
        Returns:
            Número de muestras
        """
        return len(self.data)
    
    def _construir_indice_imagenes(self) -> Dict[str, Path]:
        """
        Recorre el árbol de imágenes UNA sola vez y construye un índice
        {dicom_id: ruta_absoluta} para lookups O(1) en __getitem__.

        Usa os.scandir recursivo (mucho más rápido que pathlib.rglob) y soporta
        múltiples extensiones, priorizando .jpg.  Solo se registran archivos con
        extensión de imagen reconocida; el stem (nombre sin extensión) es el
        dicom_id.

        Returns:
            Dict {dicom_id: Path}
        """
        t0 = time.perf_counter()
        # Prioridad de extensiones: .jpg primero (no se sobrescribe por otra)
        exts_validas = {".jpg", ".jpeg", ".png"}
        indice: Dict[str, Path] = {}
        n_archivos = 0

        # Recorrido iterativo con os.scandir (rápido, sin construir Paths
        # intermedios por nivel como hace rglob).
        pendientes = [str(self.images_dir)]
        while pendientes:
            actual = pendientes.pop()
            try:
                with os.scandir(actual) as it:
                    for entry in it:
                        if entry.is_dir(follow_symlinks=False):
                            pendientes.append(entry.path)
                            continue
                        nombre = entry.name
                        punto = nombre.rfind(".")
                        if punto <= 0:
                            continue
                        ext = nombre[punto:].lower()
                        if ext not in exts_validas:
                            continue
                        n_archivos += 1
                        stem = nombre[:punto]
                        # No sobrescribir un .jpg ya indexado con otra extensión
                        if stem in indice and indice[stem].suffix.lower() == ".jpg":
                            continue
                        indice[stem] = Path(entry.path)
            except (PermissionError, FileNotFoundError):
                continue

        dt = time.perf_counter() - t0

        # Cobertura: cuántos dicom_id del CSV quedaron resueltos
        ids_csv = set(self.data["dicom_id"].astype(str))
        encontrados = sum(1 for d in ids_csv if d in indice)
        faltantes = len(ids_csv) - encontrados
        logger.info(
            "🗂️  Índice de imágenes: %d archivos indexados en %.2fs | "
            "CSV resueltos=%d/%d (faltan %d)",
            n_archivos, dt, encontrados, len(ids_csv), faltantes,
        )
        if faltantes > 0:
            logger.warning(
                "    ⚠️ %d dicom_id del CSV sin imagen en el índice → usarán placeholder",
                faltantes,
            )
        return indice

    def _buscar_imagen(self, dicom_id: str) -> Optional[Path]:
        """
        Devuelve la ruta de la imagen para un dicom_id mediante lookup O(1)
        en el índice pre-construido (sin tocar el disco).

        Args:
            dicom_id: Identificador DICOM de la imagen

        Returns:
            Path a la imagen si está en el índice, None en caso contrario
        """
        return self._indice_imagenes.get(str(dicom_id))
    
    def _mapear_vista(self, view_position: str) -> str:
        """
        Mapea el metadato de vista a categorías simples (Soft Metadata Tagging).
        
        OBJETIVO 1 - MODIFICACIÓN: Mapeo estricto de vistas:
        - 'PA' o 'AP' → 'Frontal'
        - 'LL' o 'LATERAL' → 'Lateral'
        - Otros casos → 'Unknown'
        
        Args:
            view_position: Valor original de ViewPosition del CSV
            
        Returns:
            Vista mapeada ('Frontal', 'Lateral', o 'Unknown')
        """
        if pd.isna(view_position):
            return "Unknown"
        
        view_upper = str(view_position).strip().upper()
        
        # Mapeo estricto según especificación
        if view_upper in ['PA', 'AP']:
            return "Frontal"
        elif view_upper in ['LL', 'LATERAL']:
            return "Lateral"
        else:
            return "Unknown"
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Obtiene una muestra del dataset.
        
        Este método:
        1. Carga la imagen correspondiente al índice
        2. Aplica preprocesamiento (CLAHE, padding, etc.)
        3. Extrae y mapea el metadato de vista (OBJETIVO 1)
        4. Formatea dinámicamente el prompt con contexto de vista (OBJETIVO 1)
        5. Obtiene el reporte médico asociado
        6. Tokeniza imagen y texto usando el procesador BLIP2
        7. Prepara los labels para entrenamiento
        
        Args:
            idx: Índice de la muestra a obtener
            
        Returns:
            Diccionario con claves:
                - pixel_values: Tensor de la imagen procesada
                - input_ids: IDs de tokens del texto
                - attention_mask: Máscara de atención
                - labels: Labels para el modelo (igual a input_ids)
                
        Example:
            >>> sample = dataset[0]
            >>> print(sample['pixel_values'].shape)  # torch.Size([3, 224, 224])
            >>> print(sample['input_ids'].shape)     # torch.Size([512])
        """
        # Obtener fila del DataFrame
        row = self.data.iloc[idx]
        dicom_id = row['dicom_id']
        report_text = str(row['report_text']).strip()
        
        # OBJETIVO 1 - MODIFICACIÓN: Extraer y mapear vista
        view_position = row.get('ViewPosition', None)
        vista_mapeada = self._mapear_vista(view_position)
        
        # Si el reporte está vacío, usar texto placeholder
        if not report_text:
            report_text = "chest x-ray"
            logger.warning(f"Reporte vacío para {dicom_id}, usando placeholder")
        
        # Buscar y cargar imagen
        imagen_path = self._buscar_imagen(dicom_id)

        imagen_es_placeholder = False
        if imagen_path is None or not imagen_path.exists():
            logger.warning(f"⚠️ Imagen no encontrada para {dicom_id}, usando placeholder")
            imagen_pil = crear_imagen_placeholder(self.image_size)
            imagen_es_placeholder = True
        else:
            # Procesar imagen con el pipeline completo (CLAHE → RGB → padding/resize)
            imagen_pil = estandarizar_imagen(
                imagen_path,
                target_size=self.image_size,
                clahe_clip_limit=self.clahe_clip_limit,
                clahe_tile_grid_size=self.clahe_tile_grid_size
            )
            
            # Si el procesamiento falla, usar placeholder
            if imagen_pil is None:
                logger.warning(
                    f"⚠️ Error procesando imagen {imagen_path}, usando placeholder"
                )
                imagen_pil = crear_imagen_placeholder(self.image_size)
                imagen_es_placeholder = True

        # ── Data Augmentation Espacial AL VUELO ──────────────────────────
        # DESPUÉS de CLAHE+padding y ANTES del Blip2Processor. Solo sobre imágenes
        # REALES (nunca sobre placeholders negros, donde no aportaría señal).
        augment_aplicado = False
        if self.augment and self.augment_transform is not None and not imagen_es_placeholder:
            imagen_pil = self.augment_transform(imagen_pil)
            augment_aplicado = True
        
        # OBJETIVO 2: Prompt dinámico — contexto de vista + prompt base
        # Estructura: "[Context: {vista} view] {prompt_base}"
        # Usa el cache pre-tokenizado por vista (construido en __init__) para
        # evitar reconstruir/re-tokenizar el prompt en cada muestra.
        if self.prompt is not None:
            prompt_text = self._prompt_text_cache.get(
                vista_mapeada, f"[Context: {vista_mapeada} view] {self.prompt}"
            )
        else:
            base_prompt = config.inference.default_prompt
            prompt_text = f"[Context: {vista_mapeada} view] {base_prompt}"
        
        # Tokenizar imagen y texto con el procesador de BLIP2
        try:
            # 1. Concatenamos el prompt y el reporte (Modo Causal LM estandar)
            #    + token EOS EXPLÍCITO al final del reporte para enseñar al modelo
            #    a DETENER la generación (anti Tail-Babbling). El tokenizer de OPT
            #    NO añade EOS automáticamente, por lo que sin este token el modelo
            #    nunca observa el fin de secuencia y sobre-genera en inferencia.
            eos_token = self.processor.tokenizer.eos_token or ""
            if self.prompt:
                texto_final = f"{prompt_text} {report_text}{eos_token}"
            else:
                texto_final = f"{report_text}{eos_token}"

            # 2. Procesamos todo junto
            encoding = self.processor(
                images=imagen_pil,
                text=texto_final,
                return_tensors="pt",
                padding=self.padding,
                truncation=True,
                max_length=self.max_length
            )

            # 3. Removemos la dimension extra de batch (squeeze)
            encoding = {k: v.squeeze(0) for k, v in encoding.items()}

            # 3.bis GARANTÍA EOS anti-truncación: si truncation recortó un reporte
            #    largo a max_length, el EOS del final se habría perdido. Forzamos
            #    que el último token REAL (no-pad) sea EOS para que la Cross-Entropy
            #    SIEMPRE penalice la sobre-generación, también en reportes largos.
            eos_id = self.processor.tokenizer.eos_token_id
            if eos_id is not None and "attention_mask" in encoding:
                _real = (encoding["attention_mask"] == 1).nonzero(as_tuple=True)[0]
                if len(_real) > 0:
                    _last = int(_real[-1].item())
                    if int(encoding["input_ids"][_last].item()) != eos_id:
                        encoding["input_ids"][_last] = eos_id

            # 4. CREAMOS LOS LABELS MANUALMENTE
            encoding["labels"] = encoding["input_ids"].clone()

            # 5. Enmascaramos el prompt con -100 para que el modelo solo aprenda a generar el reporte
            n_prompt_tokens = 0
            if self.prompt is not None:
                # Conteo pre-tokenizado por vista (sin llamar al tokenizer aquí)
                n_prompt_tokens = self._prompt_ntokens_cache.get(vista_mapeada)
                if n_prompt_tokens is None:
                    n_prompt_tokens = len(self.processor.tokenizer(
                        prompt_text, add_special_tokens=False)["input_ids"])
                # Ignorar la perdida en los tokens del prompt
                encoding["labels"][:n_prompt_tokens] = -100

            # ── Log detallado y ACOTADO (primeras N muestras por worker) ───────
            # Bounded por diseño: con 12-32 workers y debug_log_samples=3 son
            # ~36-96 líneas al inicio, no millones. Da trazabilidad real sin
            # inundar el log ni frenar el entrenamiento.
            if self._debug_logged < self.debug_log_samples:
                self._debug_logged += 1
                _winfo = torch.utils.data.get_worker_info()
                _wid = _winfo.id if _winfo is not None else 0
                logger.info(
                    "🔬 [muestra debug %d/%d · worker %s] idx=%d dicom=%s vista=%s",
                    self._debug_logged, self.debug_log_samples, _wid, idx, dicom_id, vista_mapeada,
                )
                logger.info("        imagen=%s | augment_aplicado=%s",
                            "placeholder" if imagen_es_placeholder else "real", augment_aplicado)
                logger.info("        prompt=%r", prompt_text)
                logger.info("        texto_final(120c)=%r", texto_final[:120])
                _eos_id = self.processor.tokenizer.eos_token_id
                _real_pos = (encoding["attention_mask"] == 1).nonzero(as_tuple=True)[0]
                _last_real = int(encoding["input_ids"][int(_real_pos[-1].item())].item()) if len(_real_pos) else -1
                logger.info(
                    "        pixel_values=%s input_ids=%s labels=%s | tokens_prompt_enmascarados=%d | ultimo_token_real=%d (EOS=%s)",
                    tuple(encoding["pixel_values"].shape),
                    tuple(encoding["input_ids"].shape),
                    tuple(encoding["labels"].shape),
                    n_prompt_tokens,
                    _last_real,
                    _last_real == _eos_id,
                )
            
            return encoding
            
        except Exception as e:
            logger.error(f"❌ Error procesando muestra {idx} (dicom_id={dicom_id}): {str(e)}")
            raise e

    def _crear_encoding_placeholder(self) -> Dict[str, torch.Tensor]:
        """
        Crea un encoding placeholder en caso de error.
        
        Returns:
            Diccionario con tensores placeholder
        """
        # Imagen placeholder (negra)
        imagen_pil = crear_imagen_placeholder(self.image_size)
        
        # Texto placeholder
        text = "chest x-ray"
        
        encoding = self.processor(
            images=imagen_pil,
            text=text,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["labels"] = encoding["input_ids"].clone()
        
        return encoding
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        Obtiene información de una muestra sin procesar la imagen completa.
        
        Útil para inspección y debugging.
        
        Args:
            idx: Índice de la muestra
            
        Returns:
            Diccionario con metadata de la muestra
        """
        row = self.data.iloc[idx]
        
        info = {
            'idx': idx,
            'dicom_id': row['dicom_id'],
            'subject_id': row.get('subject_id', 'N/A'),
            'study_id': row.get('study_id', 'N/A'),
            'report_length': len(str(row['report_text'])),
            'report_preview': str(row['report_text'])[:200] + '...'
        }
        
        return info


def crear_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True
) -> DataLoader:
    """
    Crea un DataLoader de PyTorch para el dataset.
    
    Args:
        dataset: Dataset a cargar
        batch_size: Tamaño del batch
        shuffle: Si mezclar los datos
        num_workers: Número de workers para carga paralela
        pin_memory: Si usar memoria pinneada (más rápido para GPU)
        
    Returns:
        DataLoader configurado
        
    Example:
        >>> train_loader = crear_dataloader(train_dataset, batch_size=8)
        >>> for batch in train_loader:
        ...     print(batch['pixel_values'].shape)
        ...     break
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=None  # Usaremos DataCollatorForSeq2Seq en el Trainer
    )
    
    return dataloader


# Función auxiliar para cargar splits de train/val/test
def cargar_splits(
    processor: Any,
    train_csv: Optional[Union[str, Path]] = None,
    val_csv: Optional[Union[str, Path]] = None,
    test_csv: Optional[Union[str, Path]] = None,
    images_dir: Optional[Union[str, Path]] = None
) -> Tuple[Optional[MimicCXRDataset], Optional[MimicCXRDataset], Optional[MimicCXRDataset]]:
    """
    Carga los splits de entrenamiento, validación y prueba.
    
    Args:
        processor: Procesador BLIP2
        train_csv: Ruta al CSV de entrenamiento
        val_csv: Ruta al CSV de validación
        test_csv: Ruta al CSV de prueba
        images_dir: Directorio de imágenes (usa config por defecto si no se proporciona)
        
    Returns:
        Tupla (train_dataset, val_dataset, test_dataset)
        Los datasets que no se especifiquen serán None
        
    Example:
        >>> train_ds, val_ds, test_ds = cargar_splits(
        ...     processor=processor,
        ...     train_csv="train_split_limpio.csv",
        ...     val_csv="val_split_limpio.csv"
        ... )
    """
    # Usar configuración por defecto si no se especifica
    if images_dir is None:
        images_dir = config.paths.images_dir
    
    datasets = {}
    
    for nombre, csv_path in [('train', train_csv), ('val', val_csv), ('test', test_csv)]:
        if csv_path is not None:
            try:
                logger.info(f"📂 Cargando {nombre} dataset...")
                dataset = MimicCXRDataset(
                    csv_path=csv_path,
                    images_dir=images_dir,
                    processor=processor,
                    image_size=config.data.image_size,
                    clahe_clip_limit=config.data.clahe_clip_limit,
                    clahe_tile_grid_size=config.data.clahe_tile_grid_size,
                    max_length=config.data.max_length,
                    padding=config.data.padding
                )
                datasets[nombre] = dataset
            except Exception as e:
                logger.error(f"❌ Error cargando {nombre} dataset: {str(e)}")
                datasets[nombre] = None
        else:
            datasets[nombre] = None
    
    return datasets.get('train'), datasets.get('val'), datasets.get('test')


# Ejemplo de uso
if __name__ == "__main__":
    from transformers import Blip2Processor
    
    print("🧪 Probando MimicCXRDataset...")
    
    # Cargar procesador
    print("Cargando procesador BLIP2...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    
    # Crear dataset
    print("Creando dataset...")
    dataset = MimicCXRDataset(
        csv_path=config.paths.train_csv,
        images_dir=config.paths.images_dir,
        processor=processor
    )
    
    print(f"✅ Dataset cargado con {len(dataset)} muestras")
    
    # Probar una muestra
    print("\n🔍 Probando primera muestra...")
    sample = dataset[0]
    
    print(f"  - pixel_values shape: {sample['pixel_values'].shape}")
    print(f"  - input_ids shape: {sample['input_ids'].shape}")
    print(f"  - attention_mask shape: {sample['attention_mask'].shape}")
    print(f"  - labels shape: {sample['labels'].shape}")
    
    # Mostrar info de la muestra
    print("\n📄 Información de la muestra:")
    info = dataset.get_sample_info(0)
    for key, value in info.items():
        print(f"  - {key}: {value}")
