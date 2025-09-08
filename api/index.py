# api/index.py
from contextlib import asynccontextmanager
import asyncio
import traceback
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Form, Request
from fastapi.middleware.cors import CORSMiddleware

# Config y modelos propios
from api.core.config import settings
from api.core.models import UploadResponse, QueryRequest

# Servicios propios
from api.services.ingestion import process_and_index_document
from api.services.tools import langchain_tools  # tus tools para el agente

# LangGraph
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres import PostgresSaver

# Postgres (pool síncrono)
from psycopg_pool import ConnectionPool

# LangChain "mensajes"
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

# SDK oficial OpenAI (Responses API) — compatible con sk-project…
from openai import OpenAI


# ---------------------------
# Adapter: Responses API como LLM para LangGraph/LangChain
# ---------------------------
class ResponsesLLM(Runnable):
    """Usa OpenAI Responses API como backend LLM, compatible con Project API Keys."""
    def __init__(self, model: str):
        # Toma OPENAI_API_KEY, OPENAI_ORG_ID, OPENAI_PROJECT del entorno
        self.client = OpenAI()
        self.model = model

    def _to_prompt(self, input_obj: Union[Dict[str, Any], List[Any], str]) -> str:
        # Acepta formatos típicos de LangChain/LangGraph y los convierte a texto
        if isinstance(input_obj, str):
            return input_obj

        if isinstance(input_obj, dict) and "messages" in input_obj:
            msgs = input_obj["messages"] or []
        elif isinstance(input_obj, list):
            msgs = input_obj
        else:
            # fallback: representamos todo como string
            return str(input_obj)

        parts: List[str] = []
        for m in msgs:
            if isinstance(m, HumanMessage):
                parts.append(str(m.content))
            elif isinstance(m, dict):
                # formatos tipo {"type": "human", "content": "..."} o {"content": "..."}
                t = m.get("type")
                if t == "human":
                    parts.append(str(m.get("content", "")))
                elif "content" in m and (t is None or t == "user"):
                    parts.append(str(m["content"]))
            else:
                # cualquier otro objeto
                parts.append(str(m))
        return "\n\n".join([p for p in parts if p])

    def invoke(self, input: Union[Dict[str, Any], List[Any], str], config=None, **kwargs) -> AIMessage:
        prompt = self._to_prompt(input)
        resp = self.client.responses.create(model=self.model, input=prompt)
        return AIMessage(content=resp.output_text or "")


# ---------------------------
# Lifespan: inicializa recursos en startup (no en import)
# ---------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db_pool: Optional[ConnectionPool] = None
    app.state.memory: Optional[PostgresSaver] = None
    app.state.agent = None

    # 1) Inicializa memoria (Postgres) de forma tolerante
    try:
        if getattr(settings, "POSTGRES_URI", None):
            app.state.db_pool = ConnectionPool(conninfo=settings.POSTGRES_URI)
            app.state.memory = PostgresSaver(app.state.db_pool)
            app.state.memory.setup()  # crea tablas si hace falta
    except Exception:
        # No tumbamos el server si la DB falla/está caída
        traceback.print_exc()
        app.state.memory = None

    # 2) LLM vía Responses API (compatible con sk-project…)
    llm = ResponsesLLM(model=getattr(settings, "LLM_MODEL", "gpt-4o-mini"))

    # 3) Crea el agente con o sin checkpointer según disponibilidad de DB
    app.state.agent = create_react_agent(llm, langchain_tools, checkpointer=app.state.memory)

    try:
        yield
    finally:
        try:
            if app.state.db_pool is not None:
                app.state.db_pool.close()
        except Exception:
            pass


# ---------------------------
# FastAPI app + CORS
# ---------------------------
app = FastAPI(title="IES Compliance Agent API", lifespan=lifespan)

origins = ["http://localhost:3000"]
fe = getattr(settings, "FRONTEND_URL", None)
if fe:
    origins.append(fe)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# Rutas básicas
# ---------------------------
@app.get("/")
def root():
    return {"ok": True}

@app.get("/health")
def health():
    return {"ok": True}


# ---------------------------
# Upload de documentos
# ---------------------------
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
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")


# ---------------------------
# Query del agente
# ---------------------------
@app.post("/query")
async def handle_query(query_request: QueryRequest = Body(...), request: Request = None):
    try:
        thread_id = f"tenant_{query_request.tenant_id}"
        config = {"configurable": {"thread_id": thread_id}}

        prompt_con_contexto = (
            "Eres un asistente experto en normativa de Instituciones de Educación Superior del Ecuador (IES). "
            "Tienes acceso a dos herramientas: 'simple_rag_query' y 'compliance_checklist_generator'.\n\n"
            "**Instrucciones:**\n"
            f"1) Usa siempre tenant_id='{query_request.tenant_id}'.\n"
            "2) El parámetro 'query' o 'topic' debe ser la pregunta directa del usuario.\n\n"
            f"Consulta del usuario: {query_request.query}"
        )

        input_data = {"messages": [HumanMessage(content=prompt_con_contexto)]}
        agent = request.app.state.agent
        response = await asyncio.to_thread(agent.invoke, input_data, config)

        final_msg = response.get("messages", [])[-1] if response and response.get("messages") else None
        if not final_msg:
            raise ValueError("El agente no produjo una respuesta final.")

        return {"answer": final_msg.content, "sources": []}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to execute agent query: {e}")


# ---------------------------
# Dev runner
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    # En producción usa Gunicorn; esto es para desarrollo local.
    uvicorn.run(app, host="0.0.0.0", port=8000)
