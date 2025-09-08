# api/index.py

# --- Importaciones Principales ---
import asyncio
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Form
from fastapi.middleware.cors import CORSMiddleware

# --- Importaciones de Configuración y Lógica Interna ---
from api.core.config import settings
from api.core.models import UploadResponse, QueryRequest
from api.services.ingestion import process_and_index_document
from api.services.tools import langchain_tools

# --- Importaciones de LangChain/LangGraph ---
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres import PostgresSaver # Usamos la ruta estable
from psycopg_pool import ConnectionPool

# ==============================================================================
# --- INICIALIZACIÓN DE LA APLICACIÓN ---
# ==============================================================================

app = FastAPI(title="IES Compliance Agent API con LangGraph")

# 1. Configurar CORS para permitir la comunicación con el frontend
#    Carga los orígenes permitidos desde las variables de entorno para mayor seguridad.
origins = [
    "http://localhost:3000",  # Para desarrollo local
    settings.FRONTEND_URL,    # Para producción en Vercel
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Configurar la Memoria Persistente con Postgres (de forma síncrona)
db_pool = ConnectionPool(conninfo=settings.POSTGRES_URI)
memory = PostgresSaver(db_pool)
memory.setup() # Crea las tablas de LangGraph si no existen

# 3. Crear el Agente de LangGraph una sola vez al arrancar
llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=0)
agent_executor = create_react_agent(llm, langchain_tools, checkpointer=memory)

print("✅ Servidor API y Agente inicializados correctamente.")

# ==============================================================================
# --- ENDPOINTS DE LA API ---
# ==============================================================================

@app.get("/", summary="Health Check")
def read_root():
    """Endpoint para verificar que la API está viva y funcionando."""
    return {"status": "ok", "service": "IES Compliance Agent API"}

from fastapi import BackgroundTasks
from uuid import uuid4
import asyncio

@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    tenant_id: str = Form(...),
    file: UploadFile = File(...)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    try:
        content = await file.read()
        doc_id = str(uuid4())

        # process_and_index_document es async → agenda una tarea asíncrona
        async def runner():
            await process_and_index_document(
                tenant_id=tenant_id,
                file_name=file.filename,
                file_content=content,
                document_id=doc_id  # si quieres pasarle el id generado
            )

        background_tasks.add_task(asyncio.create_task, runner())

        return UploadResponse(
            message="Document received. Processing in background.",
            document_id=doc_id,
            tenant_id=tenant_id
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@app.post("/query")
async def handle_query(query_request: QueryRequest = Body(...)):
    """Endpoint principal para interactuar con el agente RAG."""
    try:
        thread_id = f"tenant_{query_request.tenant_id}"
        config = {"configurable": {"thread_id": thread_id}}
        
        prompt_con_contexto = f"""
        Eres un asistente experto en normativa de Instituciones de Educación Superior (IES). Tu tarea es responder a la consulta del usuario.
        Tienes acceso a dos herramientas: 'simple_rag_query' y 'compliance_checklist_generator'.
        
        **Instrucciones MUY IMPORTANTES:**
        1. Para CUALQUIER herramienta que uses, DEBES pasarle el parámetro 'tenant_id'.
        2. El tenant_id para esta conversación es: '{query_request.tenant_id}'.
        3. El parámetro 'query' o 'topic' para la herramienta debe ser la pregunta directa del usuario.

        Consulta del usuario: {query_request.query}
        """
        
        input_data = {"messages": [HumanMessage(content=prompt_con_contexto)]}
        
        # Usamos asyncio.to_thread para no bloquear la API con la llamada síncrona del agente
        response = await asyncio.to_thread(
            agent_executor.invoke, input_data, config
        )
        
        final_response = response["messages"][-1]

        if not final_response:
            raise ValueError("El agente no produjo una respuesta final.")
        
        return {"answer": final_response.content, "sources": []}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to execute agent query: {str(e)}")
