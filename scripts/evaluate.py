import requests
import os
from dotenv import load_dotenv

load_dotenv()

# --- Configuración ---
API_BASE_URL = "http://127.0.0.1:8000"
TENANT_A = "cliente_universidad_xyz"
TENANT_B = "cliente_instituto_abc"
TEST_FILE_PATH = "C:/Users/Home/Downloads/Curso-RAG-Peru/Proyecto Final/documentos_prueba/doc1.pdf" # <-- CAMBIA ESTO

def run_evaluation():
    if not os.path.exists(TEST_FILE_PATH):
        print(f"❌ Archivo de prueba no encontrado en: {TEST_FILE_PATH}")
        print("    Por favor, actualiza la variable TEST_FILE_PATH en el script.")
        return

    print("🚀 Iniciando script de evaluación del RAG Agent...")

    # 1. Ingestión para el Tenant A
    print(f"\n--- 1. Ingestando documento '{os.path.basename(TEST_FILE_PATH)}' para {TENANT_A} ---")
    try:
        with open(TEST_FILE_PATH, "rb") as f:
            files = {'file': (os.path.basename(TEST_FILE_PATH), f, 'application/pdf')}
            data = {'tenant_id': TENANT_A}
            response = requests.post(f"{API_BASE_URL}/upload", files=files, data=data, timeout=120)
            response.raise_for_status()
        print(f"✅ Documento subido exitosamente. Respuesta: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error en la subida: {e}")
        return

    # 2. Consulta VÁLIDA para el Tenant A
    print(f"\n--- 2. Realizando consulta VÁLIDA para {TENANT_A} ---")
    query_a_valid = "Haz una pregunta que SÍ pueda ser respondida por tu documento." # <-- CAMBIA ESTO
    try:
        payload = {"query": query_a_valid, "tenant_id": TENANT_A}
        response = requests.post(f"{API_BASE_URL}/query", json=payload)
        response.raise_for_status()
        data = response.json()
        print(f"✅ Consulta exitosa.")
        print(f"   Respuesta: {data['answer'][:100]}...")
        print(f"   Fuentes encontradas: {len(data['sources'])}")
        assert len(data['sources']) > 0, "PRUEBA FALLIDA: Deberían encontrarse fuentes."
    except requests.exceptions.RequestException as e:
        print(f"❌ Error en la consulta: {e}")

    # 3. Consulta de AISLAMIENTO para el Tenant B
    print(f"\n--- 3. Probando aislamiento: Misma consulta desde {TENANT_B} (debe fallar) ---")
    try:
        payload = {"query": query_a_valid, "tenant_id": TENANT_B}
        response = requests.post(f"{API_BASE_URL}/query", json=payload)
        response.raise_for_status()
        data = response.json()
        print(f"✅ Consulta exitosa (como se esperaba).")
        print(f"   Respuesta: {data['answer'][:100]}...")
        print(f"   Fuentes encontradas: {len(data['sources'])}")
        assert len(data['sources']) == 0, "PRUEBA FALLIDA: ¡FUGA DE DATOS! Tenant B no debería ver datos de Tenant A."
        print("✅ PRUEBA DE AISLAMIENTO PASADA: No se encontraron fuentes.")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error en la consulta: {e}")

    print("\n🏁 Evaluación completada.")

if __name__ == "__main__":
    run_evaluation()