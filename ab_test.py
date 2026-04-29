import os
import pandas as pd
import torch
from PIL import Image, ImageFile
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from config import config

# Limpiar caché de CUDA al inicio
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Parche para imágenes ligeramente truncadas o corruptas
ImageFile.LOAD_TRUNCATED_IMAGES = True

def correr_prueba_ab():
    print("🚀 Iniciando prueba A/B de Prompts Zero-Shot...")
    
    # 1. Cargar el CSV y filtrar las que SÍ existen físicamente
    print(f"📖 Leyendo datos de: {config.paths.test_csv}")
    try:
        df_completo = pd.read_csv(config.paths.test_csv)
    except Exception as e:
        print(f"🚨 Error leyendo el CSV: {e}")
        return
    
    imagenes_existentes = []
    for _, row in df_completo.iterrows():
        img_path = config.paths.images_dir / f"{row['dicom_id']}.jpg"
        if os.path.exists(img_path):
            imagenes_existentes.append(row)
            
    if not imagenes_existentes:
        print("🚨 ALERTA: No se encontró ninguna imagen del CSV en la carpeta. Revisa la ruta.")
        return
        
    df_test = pd.DataFrame(imagenes_existentes).head(10) # Tomamos 5 que sí existen
    print(f"✅ Se encontraron {len(imagenes_existentes)} imágenes válidas. Usando las primeras {len(df_test)} para la prueba.")

    # 2. Cargar el procesador y el modelo en 16-bits (Aprovechando la RTX 3090)
    print("🧠 Cargando modelo Salesforce/blip2-opt-2.7b en FP16 (esto tomará un minuto)...")
    processor = Blip2Processor.from_pretrained(config.model.model_name)
    
    # Usar CPU si CUDA no está disponible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"📍 Usando dispositivo: {device}")
    
    model = Blip2ForConditionalGeneration.from_pretrained(
        config.model.model_name, 
        device_map="auto" if device == "cuda" else None,
        torch_dtype=dtype
    )
    if device == "cpu":
        model = model.to(device)
    
    # 3. Definir los tres prompts a probar (Estructura VQA obligatoria)
    prompt_A = "Question: Generate a medical radiology report detailing the findings for this chest X-ray. Answer:"
    prompt_B = "Question: Describe the clinical findings, anatomical structures, and abnormalities visible in this chest radiograph. Answer:"
    prompt_C = "Question: What do you see in this chest X-ray image? Describe all visible findings. Answer:"
    
    print("\n" + "="*50)
    
    # 4. Iterar sobre las 5 imágenes filtradas
    for idx, row in df_test.iterrows():
        dicom_id = row['dicom_id']
        img_path = config.paths.images_dir / f"{dicom_id}.jpg"
        
        # Obtenemos el reporte real (ajusta 'report' o 'text' según tu CSV)
        reporte_real = row.get('report_text', row.get('report', row.get('text', 'Reporte no disponible')))
        
        print(f"\n🖼️  Analizando Imagen {idx+1}/5 (ID: {dicom_id})")
        print("-" * 50)
        
        # Cargar imagen con Try/Except por seguridad
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ Error al abrir imagen {dicom_id}: {e}. Saltando...")
            continue
            
        print("👨‍⚕️ REPORTE REAL:")
        print(reporte_real[:250] + "..." if len(reporte_real) > 250 else reporte_real)
        print()
        
        # --- Generar con Prompt A (Zero-Shot puro) ---
        inputs_A = processor(image, text=prompt_A, return_tensors="pt").to("cuda", torch.float16)
        output_A = model.generate(**inputs_A, max_new_tokens=150)
        texto_A = processor.decode(output_A[0], skip_special_tokens=True).replace(prompt_A, "").strip()
        
        print(f"🤖 PROMPT A:\n{texto_A}\n")
        
        # --- Generar con Prompt B (Zero-Shot puro) ---
        inputs_B = processor(image, text=prompt_B, return_tensors="pt").to("cuda", torch.float16)
        output_B = model.generate(**inputs_B, max_new_tokens=150)
        texto_B = processor.decode(output_B[0], skip_special_tokens=True).replace(prompt_B, "").strip()
        
        print(f"🤖 PROMPT B:\n{texto_B}\n")
        
        # --- Generar con Prompt C (Zero-Shot puro) ---
        inputs_C = processor(image, text=prompt_C, return_tensors="pt").to("cuda", torch.float16)
        output_C = model.generate(**inputs_C, max_new_tokens=150)
        texto_C = processor.decode(output_C[0], skip_special_tokens=True).replace(prompt_C, "").strip()
        
        print(f"🤖 PROMPT C:\n{texto_C}")
        print("\n" + "="*50)

if __name__ == "__main__":
    correr_prueba_ab()