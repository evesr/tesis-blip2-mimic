"""
Script de evaluación e inferencia del modelo BLIP2 entrenado.

Este script permite:
- Cargar un modelo entrenado
- Generar reportes para imágenes individuales o batches
- Evaluar el modelo en el conjunto de test
- Comparar predicciones con ground truth

Author: [Tu nombre]
Date: 2026-04-22
"""

import os
import sys
import argparse
import random
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
import logging

import torch
from PIL import Image
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import config
from model_utils import cargar_modelo_entrenado, obtener_info_modelo
from preprocessing import estandarizar_imagen
from data_loader import MimicCXRDataset, cargar_splits

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GeneradorReportes:
    """
    Clase para generar reportes médicos a partir de radiografías.
    
    Encapsula el modelo entrenado y proporciona métodos convenientes
    para generación de reportes con diferentes estrategias.
    
    Attributes:
        model: Modelo BLIP2 con LoRA entrenado
        processor: Procesador BLIP2
        device: Dispositivo donde está el modelo (cuda/cpu)
        
    Example:
        >>> generador = GeneradorReportes(model_dir="modelo_entrenado")
        >>> reporte = generador.generar_desde_imagen("radiografia.jpg")
        >>> print(reporte)
    """
    
    def __init__(
        self,
        model_dir: Union[str, Path],
        model_name: str = "Salesforce/blip2-opt-2.7b",
        use_quantization: bool = True
    ):
        """
        Inicializa el generador cargando el modelo.
        
        Args:
            model_dir: Directorio con el modelo entrenado
            model_name: Nombre del modelo base
            use_quantization: Si usar cuantización 8-bit
        """
        logger.info(f"🔄 Inicializando generador de reportes...")
        
        # Cargar modelo y procesador
        self.model, self.processor = cargar_modelo_entrenado(
            model_dir=Path(model_dir),
            model_name=model_name,
            use_quantization=use_quantization
        )
        
        # Poner modelo en modo evaluación
        self.model.eval()
        
        # Detectar dispositivo
        self.device = next(self.model.parameters()).device
        logger.info(f"   - Modelo cargado en: {self.device}")
        
        # Mostrar info
        info = obtener_info_modelo(self.model)
        logger.info(f"   - Parámetros totales: {info['total_params']:,}")
        logger.info("✅ Generador listo")
    
    def generar_desde_imagen(
        self,
        imagen_path: Union[str, Path, Image.Image],
        prompt: Optional[str] = None,
        max_new_tokens: int = 150,
        num_beams: int = 5,
        temperature: float = 0.7,
        do_sample: bool = True,
        repetition_penalty: float = 1.5,
        length_penalty: float = 1.0,
        min_length: int = 30
    ) -> str:
        """
        Genera un reporte médico para una imagen.
        
        Args:
            imagen_path: Ruta a la imagen o objeto PIL Image
            prompt: Prompt para guiar la generación (usa default si es None)
            max_new_tokens: Máximo número de tokens a generar
            num_beams: Número de beams para beam search
            temperature: Temperatura para sampling
            do_sample: Si usar sampling
            repetition_penalty: Penalización por repetición
            length_penalty: Penalización/bonus por longitud
            min_length: Longitud mínima del texto generado
            
        Returns:
            Reporte generado como string
            
        Example:
            >>> reporte = generador.generar_desde_imagen(
            ...     "radiografia.jpg",
            ...     prompt="Generate detailed findings:"
            ... )
        """
        # Usar prompt por defecto si no se proporciona
        if prompt is None:
            prompt = config.inference.default_prompt
        
        # Cargar y preprocesar imagen
        if isinstance(imagen_path, (str, Path)):
            imagen = estandarizar_imagen(
                imagen_path,
                target_size=config.data.image_size,
                clahe_clip_limit=config.data.clahe_clip_limit,
                clahe_tile_grid_size=config.data.clahe_tile_grid_size
            )
            if imagen is None:
                raise ValueError(f"No se pudo procesar la imagen: {imagen_path}")
        else:
            imagen = imagen_path
        
        # Preparar inputs
        inputs = self.processor(
            images=imagen,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)
        
        # Generar con torch.no_grad() para ahorrar memoria
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_length=min_length,
                num_beams=num_beams,
                do_sample=do_sample,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                early_stopping=False
            )
        
        # Decodificar
        reporte = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()
        
        return reporte
    
    def generar_con_multiples_estrategias(
        self,
        imagen_path: Union[str, Path],
        estrategias: Optional[List[Dict]] = None
    ) -> Dict[str, str]:
        """
        Genera reportes usando múltiples estrategias de generación.
        
        Útil para comparar diferentes configuraciones de generación.
        
        Args:
            imagen_path: Ruta a la imagen
            estrategias: Lista de diccionarios con parámetros de generación
            
        Returns:
            Diccionario con nombre_estrategia: reporte_generado
            
        Example:
            >>> reportes = generador.generar_con_multiples_estrategias("xray.jpg")
            >>> for nombre, reporte in reportes.items():
            ...     print(f"{nombre}: {reporte[:100]}...")
        """
        if estrategias is None:
            # Estrategias por defecto
            estrategias = [
                {
                    'nombre': 'greedy',
                    'params': {
                        'do_sample': False,
                        'num_beams': 1,
                        'max_new_tokens': 150
                    }
                },
                {
                    'nombre': 'beam_search',
                    'params': {
                        'do_sample': False,
                        'num_beams': 5,
                        'max_new_tokens': 150
                    }
                },
                {
                    'nombre': 'sampling',
                    'params': {
                        'do_sample': True,
                        'num_beams': 1,
                        'temperature': 0.7,
                        'max_new_tokens': 150
                    }
                }
            ]
        
        resultados = {}
        
        for estrategia in estrategias:
            nombre = estrategia.get('nombre', 'unnamed')
            params = estrategia.get('params', {})
            
            logger.info(f"   Generando con estrategia: {nombre}")
            reporte = self.generar_desde_imagen(imagen_path, **params)
            resultados[nombre] = reporte
        
        return resultados
    
    def evaluar_en_dataset(
        self,
        dataset: MimicCXRDataset,
        num_samples: Optional[int] = None,
        guardar_resultados: bool = True,
        output_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Evalúa el modelo en un dataset completo.
        
        Args:
            dataset: Dataset a evaluar
            num_samples: Número de muestras a evaluar (None = todas)
            guardar_resultados: Si guardar resultados en CSV
            output_path: Ruta donde guardar resultados
            
        Returns:
            DataFrame con predicciones y ground truth
            
        Example:
            >>> resultados = generador.evaluar_en_dataset(test_dataset)
            >>> print(resultados[['dicom_id', 'predicted', 'actual']].head())
        """
        if num_samples is None:
            num_samples = len(dataset)
        else:
            num_samples = min(num_samples, len(dataset))
        
        logger.info(f"📊 Evaluando modelo en {num_samples} muestras...")
        
        resultados = []
        
        for idx in tqdm(range(num_samples), desc="Generando reportes"):
            # Obtener info de la muestra
            info = dataset.get_sample_info(idx)
            dicom_id = info['dicom_id']
            
            # Buscar imagen
            imagen_path = dataset._buscar_imagen(dicom_id)
            
            if imagen_path is None:
                logger.warning(f"Imagen no encontrada para {dicom_id}")
                continue
            
            try:
                # Generar reporte
                reporte_generado = self.generar_desde_imagen(imagen_path)
                
                # Obtener ground truth
                row = dataset.data.iloc[idx]
                reporte_real = str(row['report_text'])
                
                resultados.append({
                    'dicom_id': dicom_id,
                    'subject_id': info.get('subject_id', 'N/A'),
                    'study_id': info.get('study_id', 'N/A'),
                    'predicted_report': reporte_generado,
                    'actual_report': reporte_real,
                    'predicted_length': len(reporte_generado),
                    'actual_length': len(reporte_real)
                })
                
            except Exception as e:
                logger.error(f"Error procesando {dicom_id}: {str(e)}")
                continue
        
        # Crear DataFrame
        df_resultados = pd.DataFrame(resultados)
        
        logger.info(f"✅ Evaluación completada: {len(df_resultados)} reportes generados")
        
        # Guardar resultados
        if guardar_resultados:
            if output_path is None:
                output_path = config.paths.output_dir / "resultados_evaluacion.csv"
            
            df_resultados.to_csv(output_path, index=False)
            logger.info(f"💾 Resultados guardados en: {output_path}")
        
        return df_resultados


def visualizar_prediccion(
    imagen_path: Union[str, Path],
    reporte_generado: str,
    reporte_real: Optional[str] = None,
    save_path: Optional[Path] = None
) -> None:
    """
    Visualiza una imagen con sus reportes (generado y real).
    
    Args:
        imagen_path: Ruta a la imagen
        reporte_generado: Reporte generado por el modelo
        reporte_real: Reporte ground truth (opcional)
        save_path: Ruta donde guardar la visualización
        
    Example:
        >>> visualizar_prediccion(
        ...     "xray.jpg",
        ...     reporte_generado="Normal chest...",
        ...     reporte_real="No acute findings..."
        ... )
    """
    # Cargar imagen
    imagen = estandarizar_imagen(imagen_path)
    
    if imagen is None:
        logger.error(f"No se pudo cargar la imagen: {imagen_path}")
        return
    
    # Crear figura
    if reporte_real:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    
    # Mostrar imagen
    ax1.imshow(imagen, cmap='gray')
    ax1.axis('off')
    ax1.set_title("Radiografía", fontsize=14, fontweight='bold')
    
    # Mostrar reportes
    if reporte_real:
        # Texto de comparación
        texto_comparacion = (
            f"🤖 REPORTE GENERADO:\n"
            f"{reporte_generado[:300]}...\n\n"
            f"📄 REPORTE REAL:\n"
            f"{reporte_real[:300]}..."
        )
        
        ax2.axis('off')
        ax2.text(
            0, 1, texto_comparacion,
            va='top', ha='left',
            fontfamily='monospace',
            fontsize=9,
            wrap=True
        )
    else:
        # Solo mostrar reporte generado
        plt.text(
            0.5, -0.1,
            f"🤖 REPORTE:\n{reporte_generado}",
            ha='center', va='top',
            transform=ax1.transAxes,
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    plt.tight_layout()
    
    # Guardar o mostrar
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"💾 Visualización guardada en: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """
    Función principal para ejecutar desde línea de comandos.
    
    Example:
        # Generar reporte para una imagen
        $ python evaluate.py --model-dir modelo_entrenado --image radiografia.jpg
        
        # Evaluar en dataset de test
        $ python evaluate.py --model-dir modelo_entrenado --test-csv test_split_limpio.csv
        
        # Generar reportes aleatorios
        $ python evaluate.py --model-dir modelo_entrenado --random 5
    """
    parser = argparse.ArgumentParser(
        description="Evaluar modelo BLIP2 y generar reportes médicos"
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        required=True,
        help='Directorio con el modelo entrenado'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Ruta a imagen individual para generar reporte'
    )
    
    parser.add_argument(
        '--test-csv',
        type=str,
        help='Ruta al CSV de test para evaluación completa'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Número de muestras a evaluar del test set'
    )
    
    parser.add_argument(
        '--random',
        type=int,
        help='Generar reportes para N imágenes aleatorias del directorio'
    )
    
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Prompt personalizado para generación'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Directorio de salida para resultados'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Mostrar visualizaciones de las predicciones'
    )
    
    args = parser.parse_args()
    
    # Inicializar generador
    generador = GeneradorReportes(model_dir=args.model_dir)
    
    # Configurar output
    output_dir = Path(args.output) if args.output else config.paths.output_dir / "evaluacion"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Caso 1: Imagen individual
    if args.image:
        logger.info(f"\n🔍 Generando reporte para: {args.image}")
        
        reporte = generador.generar_desde_imagen(
            args.image,
            prompt=args.prompt
        )
        
        print("\n" + "="*80)
        print("🤖 REPORTE GENERADO:")
        print("="*80)
        print(reporte)
        print("="*80 + "\n")
        
        if args.visualize:
            visualizar_prediccion(args.image, reporte)
    
    # Caso 2: Evaluar en test set
    elif args.test_csv:
        logger.info(f"\n📊 Evaluando en dataset: {args.test_csv}")
        
        # Cargar procesador y dataset
        from model_utils import cargar_procesador
        processor = cargar_procesador()
        
        _, _, test_dataset = cargar_splits(
            processor=processor,
            test_csv=args.test_csv
        )
        
        if test_dataset is None:
            logger.error(f"No se pudo cargar el dataset: {args.test_csv}")
            return 1
        
        # Evaluar
        resultados = generador.evaluar_en_dataset(
            test_dataset,
            num_samples=args.num_samples,
            output_path=output_dir / "resultados.csv"
        )
        
        # Mostrar estadísticas
        print("\n" + "="*80)
        print("📊 ESTADÍSTICAS:")
        print("="*80)
        print(f"Muestras evaluadas: {len(resultados)}")
        print(f"Longitud promedio generada: {resultados['predicted_length'].mean():.1f}")
        print(f"Longitud promedio real: {resultados['actual_length'].mean():.1f}")
        print("="*80 + "\n")
        
        # Visualizar algunas muestras
        if args.visualize:
            logger.info("\n📸 Generando visualizaciones...")
            for idx in range(min(3, len(resultados))):
                row = resultados.iloc[idx]
                imagen_path = test_dataset._buscar_imagen(row['dicom_id'])
                if imagen_path:
                    visualizar_prediccion(
                        imagen_path,
                        row['predicted_report'],
                        row['actual_report'],
                        save_path=output_dir / f"viz_{idx}.png"
                    )
    
    # Caso 3: Imágenes aleatorias
    elif args.random:
        logger.info(f"\n🎲 Generando reportes para {args.random} imágenes aleatorias...")
        
        # Buscar imágenes en el directorio
        imagenes = list(config.paths.images_dir.rglob("*.jpg"))
        
        if not imagenes:
            logger.error(f"No se encontraron imágenes en: {config.paths.images_dir}")
            return 1
        
        # Seleccionar aleatorias
        imagenes_seleccionadas = random.sample(imagenes, min(args.random, len(imagenes)))
        
        for i, imagen_path in enumerate(imagenes_seleccionadas, 1):
            print(f"\n{'='*80}")
            print(f"Imagen {i}/{len(imagenes_seleccionadas)}: {imagen_path.name}")
            print('='*80)
            
            reporte = generador.generar_desde_imagen(imagen_path, prompt=args.prompt)
            print(reporte)
            
            if args.visualize:
                visualizar_prediccion(
                    imagen_path,
                    reporte,
                    save_path=output_dir / f"random_{i}.png"
                )
    
    else:
        parser.print_help()
        return 1
    
    logger.info("\n✅ Evaluación completada")
    return 0


if __name__ == "__main__":
    sys.exit(main())
