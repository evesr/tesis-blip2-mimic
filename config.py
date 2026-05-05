"""
Configuración del proyecto BLIP2 para generación de reportes médicos.

Este módulo contiene todas las constantes, rutas y configuraciones necesarias
para el entrenamiento y evaluación del modelo BLIP2 con LoRA para generación
automática de reportes radiológicos.

Author: [Tu nombre]
Date: 2026-04-22
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class PathConfig:
    """
    Configuración de rutas del proyecto.
    
    Attributes:
        base_dir: Directorio raíz del proyecto
        images_dir: Directorio con las imágenes de radiografías
        train_csv: Ruta al archivo CSV de entrenamiento
        val_csv: Ruta al archivo CSV de validación
        test_csv: Ruta al archivo CSV de prueba
        output_dir: Directorio para guardar modelos y checkpoints
        checkpoints_dir: Directorio específico para checkpoints durante entrenamiento
    """
    base_dir: Path = Path("/workspace/tesis-blip2-mimic/Tesis_blip2_local")
    images_dir: Path = base_dir / "imagenes_1_porciento"
    train_csv: Path = base_dir / "train_split_limpio.csv"
    val_csv: Path = base_dir / "val_split_limpio.csv"
    test_csv: Path = base_dir / "test_split_limpio.csv"
    output_dir: Path = base_dir / "modelo_entrenado"
    checkpoints_dir: Path = output_dir / "checkpoints"
    
    def create_directories(self) -> None:
        """Crea los directorios necesarios si no existen."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """
    Configuración del modelo BLIP2.
    
    Attributes:
        model_name: Nombre del modelo preentrenado en HuggingFace
        load_in_8bit: Si se debe cargar el modelo en 8-bit para ahorrar memoria
        llm_int8_threshold: Umbral para cuantización int8
        device_map: Mapa de dispositivos para distribución de modelo
    """
    model_name: str = "Salesforce/blip2-opt-2.7b"
    load_in_8bit: bool = True
    llm_int8_threshold: float = 6.0
    device_map: str = "auto"


@dataclass
class LoRAConfig:
    """
    Configuración de LoRA (Low-Rank Adaptation) para fine-tuning eficiente.
    
    Attributes:
        r: Rango de las matrices de bajo rango. Mayor = más parámetros entrenables
        lora_alpha: Factor de escalado de las actualizaciones LoRA
        lora_dropout: Dropout para regularización en capas LoRA
        bias: Tipo de bias a entrenar ("none", "all", "lora_only")
        task_type: Tipo de tarea para LoRA
        target_modules: Módulos del modelo donde aplicar LoRA
    """
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = None
    
    def __post_init__(self):
        """Inicializa target_modules si no se proporciona."""
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]


@dataclass
class TrainingConfig:
    """
    Configuración de hiperparámetros de entrenamiento.
    
    Attributes:
        per_device_train_batch_size: Tamaño de batch por GPU durante entrenamiento
        per_device_eval_batch_size: Tamaño de batch por GPU durante evaluación
        gradient_accumulation_steps: Pasos de acumulación de gradiente
        learning_rate: Tasa de aprendizaje inicial
        num_train_epochs: Número de épocas de entrenamiento
        warmup_steps: Pasos de calentamiento para el learning rate
        logging_steps: Frecuencia de logging (cada N pasos)
        save_steps: Frecuencia de guardado de checkpoints (cada N pasos)
        eval_steps: Frecuencia de evaluación (cada N pasos)
        save_strategy: Estrategia de guardado ("steps", "epoch", "no")
        eval_strategy: Estrategia de evaluación ("steps", "epoch", "no")
        save_total_limit: Número máximo de checkpoints a mantener
        load_best_model_at_end: Si cargar el mejor modelo al finalizar
        fp16: Si usar precisión mixta float16
        report_to: Dónde reportar métricas ("none", "tensorboard", "wandb")
        seed: Semilla para reproducibilidad
    """
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    num_train_epochs: int = 15
    warmup_steps: int = 100
    logging_steps: int = 50
    save_steps: Optional[int] = None
    eval_steps: Optional[int] = None
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    fp16: bool = True
    report_to: str = "none"
    seed: int = 42


@dataclass
class DataConfig:
    """
    Configuración del procesamiento de datos.
    
    Attributes:
        image_size: Tamaño de imagen de salida (ancho, alto)
        clahe_clip_limit: Límite de clip para CLAHE
        clahe_tile_grid_size: Tamaño de grid para CLAHE
        max_length: Longitud máxima de secuencia de texto
        padding: Tipo de padding ("max_length", "longest", etc.)
        train_val_split: Proporción de datos para validación (0.1 = 10%)
    """
    image_size: tuple = (224, 224)
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple = (8, 8)
    max_length: int = 512
    padding: str = "max_length"
    train_val_split: float = 0.1


@dataclass
class InferenceConfig:
    """
    Configuración para inferencia/generación de reportes.
    
    Attributes:
        max_new_tokens: Máximo número de tokens a generar
        min_length: Longitud mínima del texto generado
        num_beams: Número de beams para beam search
        do_sample: Si usar sampling durante generación
        temperature: Temperatura para sampling (mayor = más aleatorio)
        top_p: Parámetro p para nucleus sampling
        repetition_penalty: Penalización por repetición de tokens
        length_penalty: Penalización/bonificación por longitud
        early_stopping: Si detener generación anticipadamente
        default_prompt: Prompt por defecto para generación
    """
    max_new_tokens: int = 400
    min_length: int = 30
    num_beams: int = 5
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.5
    length_penalty: float = 1.0
    early_stopping: bool = False
    default_prompt: str = "Question: Generate a medical report with sections Findings and Impression based on this x-ray. Answer:"


class Config:
    """
    Configuración maestra del proyecto.
    
    Agrupa todas las configuraciones en un solo objeto para facilitar
    el acceso y la gestión de parámetros.
    """
    def __init__(self):
        self.paths = PathConfig()
        self.model = ModelConfig()
        self.lora = LoRAConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.inference = InferenceConfig()
    
    def __repr__(self) -> str:
        """Representación en string de la configuración."""
        return (
            f"Config(\n"
            f"  paths={self.paths},\n"
            f"  model={self.model},\n"
            f"  lora={self.lora},\n"
            f"  training={self.training},\n"
            f"  data={self.data},\n"
            f"  inference={self.inference}\n"
            f")"
        )


# Instancia global de configuración
config = Config()
