import tempfile
import uuid
import io
from sqlalchemy import create_engine, text
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import PagedCSVReader # You might need a specific reader
from pypdf import PdfReader # Necesitarás instalar pypdf: pip install pypdf
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from api.core.config import settings

# Cambiamos a OpenAIEmbedding para consistencia y rendimiento multilingüe
Settings.embed_model = HuggingFaceEmbedding(model_name=settings.EMBED_MODEL)
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

engine = create_engine(settings.POSTGRES_URI)

async def process_and_index_document(tenant_id: str, file_name: str, file_content: bytes) -> str:
    document_id = str(uuid.uuid4())
    
    # Leer el PDF directamente desde los bytes en memoria
    text_content = ""
    try:
        if file_name.lower().endswith('.pdf'):
            reader = PdfReader(io.BytesIO(file_content))
            for page in reader.pages:
                text_content += page.extract_text() + "\n"
        # Aquí puedes añadir lógica para otros tipos de archivo (docx, etc.)
        else:
            # Fallback para archivos de texto plano
            text_content = file_content.decode('utf-8')
    except Exception as e:
        print(f"Error parsing file {file_name}: {e}")
        # Decide si quieres lanzar una excepción o continuar con contenido vacío
        raise ValueError(f"Could not parse file: {file_name}")

    # Crear un único objeto Document de LlamaIndex
    doc = Document(
        text=text_content,
        metadata={
            "tenant_id": tenant_id,
            "document_id": document_id,
            "file_name": file_name,
        }
    )

    # Conexión a Elasticsearch
    vector_store = ElasticsearchStore(
        index_name=settings.ES_INDEX_NAME,
        es_url=settings.ES_URL,
        es_user=settings.ES_USER,
        es_password=settings.ES_PASSWORD,
    )
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Indexar el documento (LlamaIndex se encarga del chunking y embeddings)
    VectorStoreIndex.from_documents(
        [doc], # Pasamos el documento como una lista
        storage_context=storage_context,
        show_progress=True
    )
    
    # Guardar metadatos en PostgreSQL
    with engine.connect() as connection:
        stmt = text("""
            INSERT INTO documents (id, tenant_id, file_name, status)
            VALUES (:id, :tenant_id, :file_name, :status)
        """)
        connection.execute(stmt, {
            "id": document_id,
            "tenant_id": tenant_id,
            "file_name": file_name,
            "status": "INDEXED"
        })
        connection.commit()
    
    return document_id