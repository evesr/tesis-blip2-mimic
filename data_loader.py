"""
Módulo de carga de datos para el dataset MIMIC-CXR.

Implementa el Dataset de PyTorch para cargar imágenes radiológicas
y sus reportes asociados desde archivos CSV.

Author: [Tu nombre]
Date: 2026-04-22
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, Union
from PIL import Image
import logging

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
        padding: str = "max_length"
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
    
    def __len__(self) -> int:
        """
        Retorna el número total de muestras en el dataset.
        
        Returns:
            Número de muestras
        """
        return len(self.data)
    
    def _buscar_imagen(self, dicom_id: str) -> Optional[Path]:
        """
        Busca la imagen correspondiente al dicom_id en el directorio de imágenes.
        
        La búsqueda es recursiva y busca archivos .jpg que coincidan con el dicom_id.
        
        Args:
            dicom_id: Identificador DICOM de la imagen
            
        Returns:
            Path a la imagen si se encuentra, None en caso contrario
        """
        # Buscar recursivamente en el directorio de imágenes
        # Las imágenes siguen la estructura: files/p10/p10000032/s50414267/dicom_id.jpg
        
        # Intentar buscar con extensión .jpg
        imagen_path = None
        
        # Buscar recursivamente
        for archivo in self.images_dir.rglob(f"{dicom_id}.jpg"):
            imagen_path = archivo
            break
        
        if imagen_path is None:
            # Intentar con otras extensiones
            for ext in ['.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
                for archivo in self.images_dir.rglob(f"{dicom_id}{ext}"):
                    imagen_path = archivo
                    break
                if imagen_path:
                    break
        
        return imagen_path
    
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
        
        if imagen_path is None or not imagen_path.exists():
            logger.warning(f"⚠️ Imagen no encontrada para {dicom_id}, usando placeholder")
            imagen_pil = crear_imagen_placeholder(self.image_size)
        else:
            # Procesar imagen con el pipeline completo
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
        
        # OBJETIVO 1 - MODIFICACIÓN: Formatear prompt dinámicamente con contexto de vista
        # Estructura exacta: "[Context: {vista} view] Question: ... Answer: "
        prompt_text = (
            f"[Context: {vista_mapeada} view] "
            f"Question: Describe the clinical findings, anatomical structures, "
            f"and abnormalities visible in this chest radiograph. Answer: "
        )
        
        # Concatenar prompt + reporte para el formato VQA
        full_text = prompt_text + report_text
        
        # Tokenizar imagen y texto con el procesador de BLIP2
        try:
            encoding = self.processor(
                images=imagen_pil,
                text=full_text,
                padding=self.padding,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            # Remover dimensión de batch (squeeze)
            encoding = {k: v.squeeze(0) for k, v in encoding.items()}
            
            # Crear labels (copiar input_ids)
            # Durante el entrenamiento, el modelo aprenderá a predecir estos tokens
            encoding["labels"] = encoding["input_ids"].clone()
            
            return encoding
            
        except Exception as e:
            logger.error(f"❌ Error procesando muestra {idx} (dicom_id={dicom_id}): {str(e)}")
            # Retornar encoding vacío/placeholder en caso de error
            return self._crear_encoding_placeholder()
    
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
