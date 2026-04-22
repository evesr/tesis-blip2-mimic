"""
Módulo de preprocesamiento de imágenes médicas.

Contiene funciones para estandarizar y preparar imágenes radiológicas
aplicando técnicas de mejora de contraste (CLAHE) y normalización.

Author: [Tu nombre]
Date: 2026-04-22
"""

import cv2
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from typing import Optional, Tuple, Union
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def aplicar_clahe(
    imagen: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) a una imagen.
    
    CLAHE mejora el contraste local de la imagen, útil para visualizar detalles
    en radiografías donde diferentes regiones pueden tener rangos de intensidad
    muy diferentes.
    
    Args:
        imagen: Imagen en escala de grises como array numpy (H, W)
        clip_limit: Umbral para limitar el contraste. Valores más altos = más contraste
        tile_grid_size: Tamaño de las regiones para ecualización adaptativa (filas, columnas)
    
    Returns:
        Imagen con CLAHE aplicado como array numpy (H, W)
    
    Raises:
        ValueError: Si la imagen no es válida o no está en escala de grises
    
    Example:
        >>> img = cv2.imread("radiografia.jpg", cv2.IMREAD_GRAYSCALE)
        >>> img_mejorada = aplicar_clahe(img, clip_limit=2.0)
    """
    if imagen is None or imagen.size == 0:
        raise ValueError("La imagen proporcionada no es válida")
    
    if len(imagen.shape) != 2:
        raise ValueError(f"Se espera imagen en escala de grises (2D), recibido shape: {imagen.shape}")
    
    # Crear objeto CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Aplicar CLAHE
    imagen_procesada = clahe.apply(imagen)
    
    return imagen_procesada


def convertir_a_rgb(imagen: np.ndarray) -> Image.Image:
    """
    Convierte una imagen en escala de grises a formato RGB PIL.
    
    Args:
        imagen: Imagen en escala de grises como array numpy (H, W)
    
    Returns:
        Imagen PIL en formato RGB
    
    Raises:
        ValueError: Si la imagen no es válida
    
    Example:
        >>> img_gray = cv2.imread("xray.jpg", cv2.IMREAD_GRAYSCALE)
        >>> img_rgb = convertir_a_rgb(img_gray)
    """
    if imagen is None or imagen.size == 0:
        raise ValueError("La imagen proporcionada no es válida")
    
    # Convertir array numpy a imagen PIL
    imagen_pil = Image.fromarray(imagen)
    
    # Convertir a RGB (3 canales)
    imagen_rgb = imagen_pil.convert('RGB')
    
    return imagen_rgb


def aplicar_padding(
    imagen: Image.Image,
    target_size: Tuple[int, int] = (224, 224),
    fill_color: Union[str, Tuple[int, int, int]] = 'black'
) -> Image.Image:
    """
    Aplica padding a una imagen para alcanzar el tamaño objetivo manteniendo aspecto.
    
    Esta función redimensiona la imagen manteniendo la proporción de aspecto y
    agrega bordes (padding) para alcanzar exactamente el tamaño objetivo.
    
    Args:
        imagen: Imagen PIL a procesar
        target_size: Tamaño objetivo como tupla (ancho, alto)
        fill_color: Color para el padding ('black', 'white' o tupla RGB)
    
    Returns:
        Imagen PIL con padding aplicado del tamaño objetivo
    
    Example:
        >>> from PIL import Image
        >>> img = Image.open("radiografia.jpg")
        >>> img_padded = aplicar_padding(img, target_size=(224, 224))
    """
    imagen_con_padding = ImageOps.pad(
        imagen,
        target_size,
        color=fill_color,
        centering=(0.5, 0.5)  # Centrar la imagen
    )
    
    return imagen_con_padding


def estandarizar_imagen(
    ruta_imagen: Union[str, Path],
    target_size: Tuple[int, int] = (224, 224),
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
) -> Optional[Image.Image]:
    """
    Pipeline completo de preprocesamiento para una radiografía.
    
    Este es el pipeline principal que integra todos los pasos:
    1. Cargar imagen en escala de grises
    2. Aplicar CLAHE para mejorar contraste
    3. Convertir a RGB
    4. Aplicar padding para tamaño estándar
    
    Args:
        ruta_imagen: Ruta al archivo de imagen
        target_size: Tamaño objetivo de salida (ancho, alto)
        clahe_clip_limit: Límite de clip para CLAHE
        clahe_tile_grid_size: Tamaño de grid para CLAHE
    
    Returns:
        Imagen PIL procesada en RGB del tamaño objetivo, o None si falla
    
    Example:
        >>> img = estandarizar_imagen("path/to/xray.jpg")
        >>> if img is not None:
        ...     img.show()
    
    Notes:
        - Si la imagen no se puede leer, retorna None y registra un warning
        - El proceso es idempotente: aplicar dos veces no cambia el resultado
    """
    try:
        # 1. Cargar imagen en escala de grises
        img_cv = cv2.imread(str(ruta_imagen), cv2.IMREAD_GRAYSCALE)
        
        if img_cv is None:
            logger.warning(f"No se pudo leer la imagen: {ruta_imagen}")
            return None
        
        # 2. Aplicar CLAHE para mejorar contraste
        img_contrastada = aplicar_clahe(
            img_cv,
            clip_limit=clahe_clip_limit,
            tile_grid_size=clahe_tile_grid_size
        )
        
        # 3. Convertir a formato RGB
        img_rgb = convertir_a_rgb(img_contrastada)
        
        # 4. Aplicar padding para tamaño estándar
        img_final = aplicar_padding(img_rgb, target_size=target_size, fill_color='black')
        
        return img_final
        
    except Exception as e:
        logger.error(f"Error procesando imagen {ruta_imagen}: {str(e)}")
        return None


def validar_imagen(imagen: Image.Image) -> bool:
    """
    Valida que una imagen PIL tenga el formato correcto.
    
    Args:
        imagen: Imagen PIL a validar
    
    Returns:
        True si la imagen es válida, False en caso contrario
    
    Example:
        >>> img = Image.open("test.jpg")
        >>> if validar_imagen(img):
        ...     print("Imagen válida")
    """
    try:
        # Verificar que sea una imagen PIL
        if not isinstance(imagen, Image.Image):
            return False
        
        # Verificar que tenga dimensiones válidas
        width, height = imagen.size
        if width <= 0 or height <= 0:
            return False
        
        # Verificar que sea RGB o que se pueda convertir
        if imagen.mode not in ['RGB', 'L', 'RGBA']:
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validando imagen: {str(e)}")
        return False


def crear_imagen_placeholder(
    target_size: Tuple[int, int] = (224, 224),
    color: Tuple[int, int, int] = (0, 0, 0)
) -> Image.Image:
    """
    Crea una imagen placeholder (negra por defecto) del tamaño especificado.
    
    Útil cuando una imagen no se puede cargar pero necesitamos mantener
    la consistencia en el batch.
    
    Args:
        target_size: Tamaño de la imagen (ancho, alto)
        color: Color RGB de la imagen placeholder
    
    Returns:
        Imagen PIL del tamaño especificado
    
    Example:
        >>> placeholder = crear_imagen_placeholder((224, 224))
        >>> placeholder.show()
    """
    imagen = Image.new('RGB', target_size, color)
    return imagen


def procesar_batch_imagenes(
    rutas_imagenes: list[Union[str, Path]],
    target_size: Tuple[int, int] = (224, 224),
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
) -> list[Image.Image]:
    """
    Procesa un batch de imágenes aplicando el pipeline de estandarización.
    
    Args:
        rutas_imagenes: Lista de rutas a las imágenes
        target_size: Tamaño objetivo de salida
        clahe_clip_limit: Límite de clip para CLAHE
        clahe_tile_grid_size: Tamaño de grid para CLAHE
    
    Returns:
        Lista de imágenes PIL procesadas. Imágenes que fallan son reemplazadas por placeholders
    
    Example:
        >>> rutas = ["img1.jpg", "img2.jpg", "img3.jpg"]
        >>> imagenes = procesar_batch_imagenes(rutas)
        >>> print(f"Procesadas {len(imagenes)} imágenes")
    """
    imagenes_procesadas = []
    
    for ruta in rutas_imagenes:
        img = estandarizar_imagen(
            ruta,
            target_size=target_size,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_grid_size=clahe_tile_grid_size
        )
        
        # Si la imagen falla, usar placeholder
        if img is None:
            logger.warning(f"Usando placeholder para {ruta}")
            img = crear_imagen_placeholder(target_size)
        
        imagenes_procesadas.append(img)
    
    return imagenes_procesadas


# Ejemplo de uso
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Procesar imagen desde línea de comandos
        ruta = sys.argv[1]
        imagen = estandarizar_imagen(ruta)
        
        if imagen is not None:
            print(f"✅ Imagen procesada: {imagen.size}, modo: {imagen.mode}")
            imagen.show()
        else:
            print(f"❌ No se pudo procesar la imagen: {ruta}")
    else:
        print("Uso: python preprocessing.py <ruta_imagen>")
        print("Ejemplo: python preprocessing.py radiografia.jpg")
