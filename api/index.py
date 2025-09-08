from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Form, Request
from api.core.config import settings # Importa settings para cargar variables de entorno
from api.core.models import UploadResponse, QueryRequest, QueryResponse
from api.services.ingestion import process_and_index_document
from api.services.query import query_index
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.openai import OpenAI
from api.services.query import get_query_engine_for_tenant # Importamos una función factorizada
from fastapi import FastAPI, HTTPException, Body
from api.core.config import settings
from api.core.models import QueryRequest, QueryResponse, SourceNode
from contextlib import asynccontextmanager



# --- Importaciones de LangChain/LangGraph ---
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
import traceback # <-- Añade esta importación al principio del archivo
from api.services.tools import langchain_tools
from psycopg_pool import AsyncConnectionPool
import asyncio
from fastapi.middleware.cors import CORSMiddleware
# Eliminamos el 'lifespan' y volvemos a la inicialización directa y síncrona
app = FastAPI(title="IES Compliance Agent API con LangGraph")
# 2. Configura el middleware de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # El origen de tu app Next.js en desarrollo
    allow_credentials=True,
    allow_methods=["*"], # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"], # Permite todas las cabeceras
)
# Paso 1: Crear un pool de conexiones SÍNCRONO, como en tu app original
db_pool = ConnectionPool(conninfo=settings.POSTGRES_URI)

# Paso 2: Pasar el pool directamente al constructor de PostgresSaver
memory = PostgresSaver(db_pool)

# Paso 3: Llamar a .setup() directamente. Esto sí funciona en la versión síncrona.
memory.setup()

# --- FIN DE LA CORRECCIÓN FINAL ---

# --- Creación del Agente de LangGraph (esto no cambia) ---
llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=0)
agent_executor = create_react_agent(llm, langchain_tools, checkpointer=memory)




# --- Configuración de la Memoria Persistente (PostgresSaver) ---


@app.post("/upload", response_model=UploadResponse)
async def upload_document(tenant_id: str = Form(...), file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    
    try:
        file_content = await file.read()
        document_id = await process_and_index_document(
            tenant_id=tenant_id,
            file_name=file.filename,
            file_content=file_content
        )
        return UploadResponse(
            message="Document indexed successfully",
            document_id=document_id,
            tenant_id=tenant_id
        )
    except Exception as e:
        # Idealmente, aquí habría un logger más robusto
        print(f"ERROR during upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@app.post("/query")
async def handle_query(query_request: QueryRequest = Body(...)):
    try:
        thread_id = f"tenant_{query_request.tenant_id}"
        config = {"configurable": {"thread_id": thread_id}}

        # --- INICIO DEL PROMPT MEJORADO ---
        # Le damos instrucciones mucho más claras al agente sobre cómo debe comportarse
        # y cómo usar los parámetros de las herramientas.
        prompt_con_contexto = f"""
        Eres un asistente experto en normativa de Instituciones de Eduación Superior del Ecuador (IES). Tu tarea es responder a la consulta del usuario.
        Tienes acceso a dos herramientas: 'simple_rag_query' y 'compliance_checklist_generator'.
        
        **Instrucciones MUY IMPORTANTES:**
        1. Para CUALQUIER herramienta que uses, DEBES pasarle el parámetro 'tenant_id'.
        2. El tenant_id para esta conversación es: '{query_request.tenant_id}'.
        3. El parámetro 'query' o 'topic' para la herramienta debe ser la pregunta directa del usuario.

        Consulta del usuario: {query_request.query}
        """
        # --- FIN DEL PROMPT MEJORADO ---
        
        input_data = {"messages": [HumanMessage(content=prompt_con_contexto)]}
        
        response = await asyncio.to_thread(
            agent_executor.invoke, input_data, config
        )
        
        final_response = response["messages"][-1]

        if not final_response:
            raise ValueError("El agente no produjo una respuesta final.")
        
        return {"answer": final_response.content, "sources": []}

    except Exception as e:
        import traceback
        print("\n--- INICIO DEL TRACEBACK DETALLADO ---")
        traceback.print_exc()
        print("--- FIN DEL TRACEBACK DETALLADO ---\n")
        raise HTTPException(status_code=500, detail=f"Failed to execute agent query: {str(e)}")