"""
Utilidades para manejo de modelos BLIP2 con LoRA.

Este módulo proporciona funciones para cargar, configurar y guardar modelos BLIP2
con fine-tuning eficiente usando LoRA (Low-Rank Adaptation).

Author: [Tu nombre]
Date: 2026-04-22
"""

import torch
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging

from config import config

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cargar_procesador(
    model_name: str = "Salesforce/blip2-opt-2.7b",
    cache_dir: Optional[str] = None
) -> Blip2Processor:
    """
    Carga el procesador de BLIP2 para tokenización de imágenes y texto.
    
    El procesador se encarga de:
    - Preparar imágenes para el vision encoder
    - Tokenizar texto para el language model
    
    Args:
        model_name: Nombre del modelo en HuggingFace Hub
        cache_dir: Directorio para cachear el procesador descargado
        
    Returns:
        Procesador BLIP2 listo para usar
        
    Example:
        >>> processor = cargar_procesador()
        >>> inputs = processor(images=image, text="caption", return_tensors="pt")
    """
    logger.info(f"📥 Cargando procesador BLIP2 desde {model_name}...")
    
    processor = Blip2Processor.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    
    logger.info("✅ Procesador cargado exitosamente")
    return processor


def crear_configuracion_cuantizacion(
    load_in_8bit: bool = True,
    llm_int8_threshold: float = 6.0
) -> BitsAndBytesConfig:
    """
    Crea configuración para cuantización del modelo (8-bit).
    
    La cuantización reduce el uso de memoria permitiendo cargar modelos grandes
    en GPUs con memoria limitada, con mínima pérdida de rendimiento.
    
    Args:
        load_in_8bit: Si cargar el modelo en precisión de 8 bits
        llm_int8_threshold: Umbral para outlier features en cuantización
        
    Returns:
        Configuración de cuantización para BitsAndBytes
        
    Example:
        >>> bnb_config = crear_configuracion_cuantizacion(load_in_8bit=True)
        >>> model = Blip2ForConditionalGeneration.from_pretrained(
        ...     "Salesforce/blip2-opt-2.7b",
        ...     quantization_config=bnb_config
        ... )
        
    Notes:
        - 8-bit reduce uso de memoria ~50% con <1% pérdida de precisión
        - Requiere GPU con soporte para INT8 (Turing o más reciente)
    """
    logger.info(f"⚙️ Creando configuración de cuantización (8-bit: {load_in_8bit})...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=load_in_8bit,
        llm_int8_threshold=llm_int8_threshold
    )
    
    return bnb_config


def cargar_modelo_base(
    model_name: str = "Salesforce/blip2-opt-2.7b",
    use_quantization: bool = True,
    device_map: str = "auto",
    cache_dir: Optional[str] = None
) -> Blip2ForConditionalGeneration:
    """
    Carga el modelo base BLIP2 (sin LoRA).
    
    Args:
        model_name: Nombre del modelo en HuggingFace Hub
        use_quantization: Si usar cuantización 8-bit
        device_map: Estrategia de mapeo a dispositivos ("auto", "cuda", "cpu")
        cache_dir: Directorio para cachear el modelo descargado
        
    Returns:
        Modelo BLIP2 base cargado
        
    Example:
        >>> model = cargar_modelo_base()
        >>> print(f"Modelo cargado en: {model.device}")
        
    Notes:
        - El modelo base tiene ~2.7B parámetros
        - Con cuantización ocupa ~2.7GB de VRAM
        - Sin cuantización ocupa ~5.4GB de VRAM
    """
    logger.info(f"⏳ Cargando modelo base BLIP2 desde {model_name}...")
    logger.info("   (Esto puede tardar varios minutos en la primera carga)")
    
    # Configurar cuantización si se solicita
    quantization_config = None
    if use_quantization:
        quantization_config = crear_configuracion_cuantizacion(
            load_in_8bit=config.model.load_in_8bit,
            llm_int8_threshold=config.model.llm_int8_threshold
        )
    
    # Cargar modelo
    # torch_dtype: bfloat16 en todos los casos.
    # - Mismo rango de exponente que float32 -> sin overflow ni underflow.
    # - Misma huella de memoria que float16 (2 bytes por parametro).
    # - RTX 3090 (Ampere) tiene soporte nativo de bfloat16.
    # - Con bf16=True en el Trainer no se instancia GradScaler,
    #   eliminando el crash 'Attempting to unscale FP16 gradients'.
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map if device_map else "auto",
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir
    )
    
    logger.info("✅ Modelo base cargado exitosamente")
    
    # Verificar que esté en GPU
    if torch.cuda.is_available():
        device = next(model.parameters()).device
        logger.info(f"📍 Modelo cargado en: {device}")
    
    return model


def crear_configuracion_lora(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
    target_modules: Optional[list] = None
) -> LoraConfig:
    """
    Crea configuración de LoRA para fine-tuning eficiente.
    
    LoRA permite entrenar modelos grandes añadiendo pequeñas matrices de bajo rango
    a las capas del modelo, reduciendo dramáticamente los parámetros entrenables.
    
    Args:
        r: Rango de las matrices LoRA (mayor = más capacidad, más parámetros)
        lora_alpha: Factor de escalado para las actualizaciones LoRA
        lora_dropout: Dropout en las capas LoRA para regularización
        bias: Qué bias entrenar ("none", "all", "lora_only")
        task_type: Tipo de tarea ("CAUSAL_LM" para generación de texto)
        target_modules: Lista de nombres de módulos donde aplicar LoRA
        
    Returns:
        Configuración LoRA lista para aplicar al modelo
        
    Example:
        >>> lora_config = crear_configuracion_lora(r=16, lora_alpha=32)
        >>> model = get_peft_model(base_model, lora_config)
        
    Notes:
        - r=16, alpha=32 son buenos valores por defecto
        - target_modules típicamente son capas de atención: ["q_proj", "v_proj"]
        - Dropout 0.05-0.1 ayuda a prevenir overfitting
    """
    if target_modules is None:
        target_modules = config.lora.target_modules
    
    logger.info(f"⚙️ Creando configuración LoRA (r={r}, alpha={lora_alpha})...")
    
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
        target_modules=target_modules
    )
    
    return lora_config


def aplicar_lora_a_modelo(
    model: Blip2ForConditionalGeneration,
    lora_config: Optional[LoraConfig] = None
) -> Blip2ForConditionalGeneration:
    """
    Aplica configuración LoRA a un modelo base.
    
    Esto congela los parámetros del modelo base y añade capas LoRA entrenables.
    
    Args:
        model: Modelo base BLIP2
        lora_config: Configuración LoRA (usa config por defecto si es None)
        
    Returns:
        Modelo con LoRA aplicado
        
    Example:
        >>> base_model = cargar_modelo_base()
        >>> model_lora = aplicar_lora_a_modelo(base_model)
        >>> model_lora.print_trainable_parameters()
        
    Notes:
        - Solo se entrenan ~1% de los parámetros con LoRA
        - El modelo resultante mantiene el mismo rendimiento que full fine-tuning
    """
    if lora_config is None:
        lora_config = crear_configuracion_lora(
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
            bias=config.lora.bias,
            task_type=config.lora.task_type,
            target_modules=config.lora.target_modules
        )
    
    logger.info("🔧 Aplicando LoRA al modelo...")
    model_peft = get_peft_model(model, lora_config)
    
    # Mostrar estadísticas de parámetros
    logger.info("\n📊 ESTADÍSTICAS DEL MODELO:")
    model_peft.print_trainable_parameters()
    
    return model_peft


def cargar_modelo_completo(
    use_lora: bool = True,
    model_name: str = "Salesforce/blip2-opt-2.7b",
    use_quantization: bool = True
) -> Blip2ForConditionalGeneration:
    """
    Carga modelo base y opcionalmente aplica LoRA.
    
    Esta es la función principal para preparar el modelo para entrenamiento.
    
    Args:
        use_lora: Si aplicar LoRA al modelo
        model_name: Nombre del modelo en HuggingFace
        use_quantization: Si usar cuantización 8-bit
        
    Returns:
        Modelo listo para entrenar o realizar inferencia
        
    Example:
        >>> # Para entrenamiento
        >>> model = cargar_modelo_completo(use_lora=True)
        >>> 
        >>> # Solo para inferencia (sin LoRA)
        >>> model = cargar_modelo_completo(use_lora=False)
    """
    # Cargar modelo base
    model = cargar_modelo_base(
        model_name=model_name,
        use_quantization=use_quantization,
        device_map=config.model.device_map
    )
    
    # Aplicar LoRA si se solicita
    if use_lora:
        model = aplicar_lora_a_modelo(model)
    
    return model


def guardar_modelo(
    model: Blip2ForConditionalGeneration,
    processor: Blip2Processor,
    output_dir: Path,
    save_full_model: bool = False
) -> None:
    """
    Guarda el modelo y el procesador en disco.
    
    Args:
        model: Modelo a guardar (con o sin LoRA)
        processor: Procesador a guardar
        output_dir: Directorio donde guardar
        save_full_model: Si guardar modelo completo o solo adaptadores LoRA
        
    Example:
        >>> guardar_modelo(model, processor, Path("./modelo_entrenado"))
        
    Notes:
        - Si el modelo tiene LoRA, por defecto solo guarda los adaptadores (~100MB)
        - Con save_full_model=True guarda todo el modelo (~2.7GB)
        - Siempre guarda el procesador para reproducibilidad
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"💾 Guardando modelo en: {output_dir}")
    
    # Guardar modelo
    if save_full_model:
        logger.info("   - Guardando modelo completo...")
        model.save_pretrained(output_dir)
    else:
        # Solo guardar adaptadores LoRA (mucho más pequeño)
        logger.info("   - Guardando adaptadores LoRA...")
        try:
            model.save_pretrained(output_dir)
        except AttributeError:
            # Si no es un modelo PEFT, guardar completo
            logger.warning("   - Modelo no tiene LoRA, guardando modelo completo...")
            model.save_pretrained(output_dir)
    
    # Guardar procesador
    logger.info("   - Guardando procesador...")
    processor.save_pretrained(output_dir)
    
    logger.info(f"✅ Modelo guardado exitosamente en: {output_dir}")


def cargar_modelo_entrenado(
    model_dir: Path,
    model_name: str = "Salesforce/blip2-opt-2.7b",
    use_quantization: bool = True
) -> Tuple[Blip2ForConditionalGeneration, Blip2Processor]:
    """
    Carga un modelo previamente entrenado con LoRA.
    
    Args:
        model_dir: Directorio donde está guardado el modelo
        model_name: Nombre del modelo base original
        use_quantization: Si usar cuantización al cargar
        
    Returns:
        Tupla (modelo, procesador) listos para inferencia
        
    Example:
        >>> model, processor = cargar_modelo_entrenado(Path("./modelo_entrenado"))
        >>> # Ahora puedes usar el modelo para generar reportes
        
    Notes:
        - Primero carga el modelo base
        - Luego aplica los adaptadores LoRA guardados
        - Esto permite reutilizar el modelo base y solo guardar las "gafas" LoRA
    """
    model_dir = Path(model_dir)
    
    logger.info(f"📂 Cargando modelo entrenado desde: {model_dir}")
    
    # 1. Cargar procesador
    logger.info("   - Cargando procesador...")
    processor = Blip2Processor.from_pretrained(model_dir)
    
    # 2. Cargar modelo base
    logger.info("   - Cargando modelo base...")
    base_model = cargar_modelo_base(
        model_name=model_name,
        use_quantization=use_quantization
    )
    
    # 3. Aplicar adaptadores LoRA
    logger.info("   - Aplicando adaptadores LoRA...")
    try:
        model = PeftModel.from_pretrained(base_model, model_dir)
        logger.info("✅ Modelo con LoRA cargado exitosamente")
    except Exception as e:
        logger.warning(f"   - No se encontraron adaptadores LoRA: {e}")
        logger.info("   - Usando modelo base sin LoRA")
        model = base_model
    
    return model, processor


def obtener_info_modelo(model: Blip2ForConditionalGeneration) -> Dict[str, Any]:
    """
    Obtiene información sobre un modelo BLIP2.
    
    Args:
        model: Modelo a analizar
        
    Returns:
        Diccionario con información del modelo
        
    Example:
        >>> info = obtener_info_modelo(model)
        >>> print(f"Parámetros totales: {info['total_params']:,}")
        >>> print(f"Parámetros entrenables: {info['trainable_params']:,}")
    """
    info = {}
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info['total_params'] = total_params
    info['trainable_params'] = trainable_params
    info['trainable_percentage'] = 100 * trainable_params / total_params if total_params > 0 else 0
    info['frozen_params'] = total_params - trainable_params
    
    # Tipo de modelo
    info['model_type'] = model.__class__.__name__
    info['has_lora'] = hasattr(model, 'peft_config')
    
    # Dispositivo
    info['device'] = str(next(model.parameters()).device)
    info['dtype'] = str(next(model.parameters()).dtype)
    
    return info


def imprimir_info_modelo(model: Blip2ForConditionalGeneration) -> None:
    """
    Imprime información detallada sobre el modelo.
    
    Args:
        model: Modelo a analizar
        
    Example:
        >>> imprimir_info_modelo(model)
    """
    info = obtener_info_modelo(model)
    
    print("\n" + "="*60)
    print("📊 INFORMACIÓN DEL MODELO")
    print("="*60)
    print(f"Tipo: {info['model_type']}")
    print(f"Tiene LoRA: {'✅ Sí' if info['has_lora'] else '❌ No'}")
    print(f"Dispositivo: {info['device']}")
    print(f"Tipo de dato: {info['dtype']}")
    print(f"\nParámetros totales: {info['total_params']:,}")
    print(f"Parámetros entrenables: {info['trainable_params']:,}")
    print(f"Parámetros congelados: {info['frozen_params']:,}")
    print(f"Porcentaje entrenable: {info['trainable_percentage']:.2f}%")
    print("="*60 + "\n")


# Ejemplo de uso
if __name__ == "__main__":
    print("🧪 Probando utilidades de modelo...\n")
    
    # 1. Cargar procesador
    processor = cargar_procesador()
    
    # 2. Cargar modelo base con LoRA
    print("\n" + "-"*60)
    model = cargar_modelo_completo(use_lora=True)
    
    # 3. Mostrar información
    imprimir_info_modelo(model)
    
    # 4. Simular guardado
    print("\n" + "-"*60)
    output_test = Path("./test_modelo_output")
    guardar_modelo(model, processor, output_test, save_full_model=False)
    
    print("\n✅ Todas las funciones probadas exitosamente")
