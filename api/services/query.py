from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from api.core.config import settings
from api.core.models import QueryResponse, SourceNode

Settings.llm = OpenAI(model=settings.LLM_MODEL)
# El embed_model ya está configurado en ingestion.py

def get_query_engine_for_tenant(tenant_id: str):
    vector_store = ElasticsearchStore(
        index_name=settings.ES_INDEX_NAME,
        es_url=settings.ES_URL,
        es_user=settings.ES_USER,
        es_password=settings.ES_PASSWORD,
    )
    
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    # --- INICIO DE LA LÓGICA MODIFICADA ---
    # Creamos el filtro para buscar en los documentos del tenant actual Y en los globales
    filters = MetadataFilters(
        filters=[
            ExactMatchFilter(key="tenant_id", value=tenant_id),
            ExactMatchFilter(key="tenant_id", value=settings.GLOBAL_TENANT_ID),
        ],
        condition="or"  # <--- Condición "OR" para que busque en cualquiera de los dos
    )
    query_engine = index.as_query_engine(
        filters=filters,
        similarity_top_k=5 # Aumentamos un poco el top_k para tener más contexto de ambas fuentes
    )
    return query_engine

def query_index(query: str, tenant_id: str) -> QueryResponse: # Quitamos async
    query_engine = get_query_engine_for_tenant(tenant_id)
    response = query_engine.query(query) # Cambiamos a .query() síncrono
    
    sources = [
        SourceNode(...) for node in response.source_nodes
    ]
    
    return QueryResponse(answer=str(response), sources=sources)