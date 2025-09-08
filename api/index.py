# api/index.py
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Form, Request
from fastapi.middleware.cors import CORSMiddleware

from api.core.config import settings
from api.core.models import UploadResponse, QueryRequest  # QueryResponse si lo necesitas
from api.services.ingestion import process_and_index_document
from api.services.tools import langchain_tools

# LangGraph / LangChain
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres import PostgresSaver

# Postgres (sync)
from psycopg_pool import ConnectionPool

import asyncio
import traceback


# ---------- Lifespan: inicializa y cierra recursos ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1) Pool de conexiones (SYNC)
    try:
        # Asegura que tu POSTGRES_URI use sslmode=require si es DB externa
        app.state.db_pool = ConnectionPool(conninfo=settings.POSTGRES_URI)
    except Exception:
        traceback.print_exc()
        raise

    # 2) Checkpointer de LangGraph (SYNC)
    try:
        app.state.memory = PostgresSaver(app.state.db_pool)
        app.state.memory.setup()  # sync setup
    except Exception:
        traceback.print_exc()
        # Cierra el pool si falló
        try:
            app.state.db_pool.close()
        except Exception:
            pass
        raise

    # 3) LLM y agente (ligeros pero mejor aquí que al importar)
    try:
        llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=0)
        app.state.agent_executor = create_react_agent(
            llm,
            langchain_tools,
            checkpointer=app.state.memory
        )
    except Exception:
        traceback.print_exc()
        try:
            app.state.db_pool.close()
        except Exception:
            pass
        raise

    # listo para servir
    yield

    # ---------- shutdown ----------
    try:
        app.state.db_pool.close()
    except Exception:
        pass


app = FastAPI(title="IES Compliance Agent API con LangGraph", lifespan=lifespan)

# ---------- CORS ----------
origins = ["http://localhost:3000"]
if getattr(settings, "FRONTEND_URL", None):
    origins.append(settings.FRONTEND_URL)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Health ----------
@app.get("/health")
def health():
    return {"ok": True}

# ---------- Endpoints ----------
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
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@app.post("/query")
async def handle_query(query_request: QueryRequest = Body(...), request: Request = None):
    try:
        thread_id = f"tenant_{query_request.tenant_id}"
        config = {"configurable": {"thread_id": thread_id}}

        prompt_con_contexto = (
            "Eres un asistente experto en normativa de Instituciones de Educación Superior del Ecuador (IES). "
            "Tienes acceso a dos herramientas: 'simple_rag_query' y 'compliance_checklist_generator'.\n\n"
            "Instrucciones:\n"
            "1) Siempre pasa 'tenant_id' a cualquier herramienta que uses.\n"
            f"2) El tenant_id para esta conversación es: '{query_request.tenant_id}'.\n"
            "3) El parámetro 'query' o 'topic' debe ser la pregunta directa del usuario.\n\n"
            f"Consulta del usuario: {query_request.query}"
        )

        input_data = {"messages": [HumanMessage(content=prompt_con_contexto)]}
        agent = request.app.state.agent_executor

        # 'invoke' es sincrónico; correr en hilo para no bloquear el loop
        response = await asyncio.to_thread(agent.invoke, input_data, config)
        final_msg = response.get("messages", [])[-1] if response.get("messages") else None
        if not final_msg:
            raise ValueError("El agente no produjo una respuesta final.")

        return {"answer": final_msg.content, "sources": []}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to execute agent query: {str(e)}")


# Para ejecución directa (desarrollo)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
