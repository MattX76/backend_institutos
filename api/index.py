# api/index.py
import asyncio
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Form
from fastapi.middleware.cors import CORSMiddleware

from api.core.config import settings
from api.core.models import UploadResponse, QueryRequest
from api.services.ingestion import process_and_index_document # Asegúrate de que esta función sea asíncrona
from api.services.tools import langchain_tools

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.psql import PostgresSaver
from psycopg_pool import ConnectionPool

# --- INICIALIZACIÓN DE LA APP Y CORS ---
app = FastAPI(title="IES Compliance Agent API con LangGraph")

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

# --- INICIALIZACIÓN SÍNCRONA DEL AGENTE ---
db_pool = ConnectionPool(conninfo=settings.POSTGRES_URI)
memory = PostgresSaver(db_pool)
memory.setup()

llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=0)
agent_executor = create_react_agent(llm, langchain_tools, checkpointer=memory)

# --- ENDPOINTS ---
@app.get("/", summary="Health Check")
def read_root():
    return {"status": "ok", "service": "IES Compliance Agent API"}

@app.post("/upload", response_model=UploadResponse)
async def upload_document(tenant_id: str = Form(...), file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    try:
        file_content = await file.read()
        document_id = await process_and_index_document( # La ingestión puede ser intensiva, la dejamos asíncrona
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
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@app.post("/query")
async def handle_query(query_request: QueryRequest = Body(...)):
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
