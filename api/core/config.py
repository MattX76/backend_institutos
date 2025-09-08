# api/core/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra='ignore')

    # --- ASEGÚRATE DE QUE ESTAS LÍNEAS ESTÉN PRESENTES ---
    # Modelos y APIs
    EMBED_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # Modelo de Hugging Face

    LLM_MODEL: str = "gpt-4o-mini"
    OPENAI_API_KEY: str
    
    # LangSmith Tracing
    LANGCHAIN_API_KEY: str
    LANGCHAIN_TRACING_V2: str = "true"
    LANGCHAIN_PROJECT: str

    # Infraestructura (versión con URL)
    ES_URL: str
    ES_USER: str
    ES_PASSWORD: str
    POSTGRES_URI: str
    
    # Aplicación
    ES_INDEX_NAME: str = "ies_normativa_multi_tenant"
    GLOBAL_TENANT_ID: str = "dueño_del_servicio_global" # <-- AÑADE ESTA LÍNEA
    FRONTEND_URL: str
settings = Settings()
print(f"--- URI LEÍDA POR LA APP: {settings.POSTGRES_URI} ---")

# Configurar variables de entorno para LangSmith y OpenAI
os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = settings.LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT