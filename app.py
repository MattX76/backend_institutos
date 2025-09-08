# -*- coding: utf-8 -*-
import os
import tempfile
from pathlib import Path
from typing import List, Tuple
import uuid, datetime as dt, re, json
import pandas as pd
from sqlalchemy import create_engine, text
import json
import datetime as dt
import pandas as pd
import streamlit as st

# LangChain / LangGraph
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
#Elastic search
from elasticsearch import Elasticsearch as ESClient
import sys, asyncio
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# LlamaIndex
from llama_index.core import (
    SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings, Document
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI as LI_OpenAI
from llama_index.vector_stores.elasticsearch import ElasticsearchStore

import re

# Índice base por defecto (el global que ya usabas)
ES_INDEX_BASE = os.getenv("ES_INDEX", "rag_llama")

def _slug(s: str) -> str:
    # Reglas ES: minúsculas; permitidos a-z 0-9 _ - +; no empezar con - _ +
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9_\-+]+", "-", s).strip("-_+.")
    if not s or s[0] in "-_+":
        s = f"x-{s}"
    return s[:200]  # margen para no pasar el límite

def compute_es_index(strategy: str, user_name: str | None, thread_id: str | None) -> str:
    if strategy == "Por usuario" and user_name:
        return f"{_slug(ES_INDEX_BASE)}--u--{_slug(user_name)}"
    if strategy == "Por hilo" and thread_id:
        return f"{_slug(ES_INDEX_BASE)}--t--{_slug(thread_id)}"
    return _slug(ES_INDEX_BASE)


# ─────────────────────────── Helpers para cargar secretos ───────────────────────────
def _read_file(path: str) -> str | None:
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except Exception:
        return None

# 1) Postgres (memoria)
PG_DSN = _read_file("C:/Users/Home/Downloads/Curso-RAG-Peru/Sesion 30-20250821/proyecto/secrets/postgrest.txt") or os.getenv("PG_DSN")

# 2) OpenAI API
OPENAI_API_KEY = _read_file("C:/Users/Home/Downloads/Curso-RAG-Peru/Sesion 30-20250821/proyecto/secrets/api_key.txt") or os.getenv("OPENAI_API_KEY")

# 3) Elasticsearch password
ES_PASSWORD_FILE = _read_file("C:/Users/Home/Downloads/Curso-RAG-Peru/Sesion 30-20250821/proyecto/secrets/elasticstore.txt")
ES_PASSWORD = ES_PASSWORD_FILE or os.getenv("ES_PASSWORD")

# 4) LangGraph/LangSmith tracing (opcional)
LG_API = _read_file("C:/Users/Home/Downloads/Curso-RAG-Peru/Sesion 30-20250821/proyecto/secrets/langgraphapi.txt")
if LG_API:
    os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    os.environ["LANGCHAIN_API_KEY"] = LG_API
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "rag-streamlit")

# Otros parámetros (puedes cambiarlos por entorno si quieres)
ES_URL       = os.getenv("ES_URL", "http://35.238.61.206:9200")
ES_USER      = os.getenv("ES_USER", "elastic")
ES_INDEX     = os.getenv("ES_INDEX", "rag_llama")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ───────────────────────────────── Config & checks ─────────────────────────────────
st.set_page_config(page_title="RAG (LlamaIndex + LangGraph)", layout="wide")
st.title("🧠 RAG con LlamaIndex + LangGraph (ES + Postgres)")
st.caption("Sube archivos → se indexan en Elasticsearch → consulta con agente ReAct (memoria en Postgres).")

if not OPENAI_API_KEY:
    st.error("❌ Falta OPENAI_API_KEY (/api_key.txt o variable de entorno).")
    st.stop()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

if not ES_PASSWORD:
    st.error("❌ Falta contraseña de Elasticsearch (/elasticstore.txt o variable ES_PASSWORD).")
    st.stop()

if not PG_DSN:
    st.error("❌ Falta DSN de Postgres (/postgrest.txt o variable PG_DSN).")
    st.stop()

# Detecta GPU opcional para embeddings
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

# ─── Crear tablas si no existen + inicializar memoria ─────────────────────────
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver

# Si tu variable se llama 'uribd' porque la leíste de archivo, usa:
# PG_DSN = uribd

connection_kwargs = {"autocommit": True, "prepare_threshold": 0}

@st.cache_resource
def init_memory(dsn: str):
    """Crea pool, checkpointer y asegura tablas de memoria (idempotente)."""
    pool = ConnectionPool(conninfo=dsn, max_size=20, kwargs=connection_kwargs)
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()  # crea checkpoints, checkpoint_blobs, (checkpoint_)writes, checkpoint_migrations
    return pool, checkpointer

st.session_state.pg_pool, st.session_state.checkpointer = init_memory(PG_DSN)

import re, uuid, datetime as dt
import streamlit as st

def slugify(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')

st.sidebar.markdown("### 🧑 Identidad & sesión")

user_name = st.sidebar.text_input("Tu nombre o ID", value=st.session_state.get("user_name", ""), placeholder="p.ej. ana.soto")
if user_name:
    st.session_state.user_name = user_name

mode = st.sidebar.radio("Hilo de conversación", ["Continuar por nombre", "Crear nuevo hilo"], index=0)

# Opcional: listar hilos existentes desde Postgres si quieres reanudar exactamente uno:
existing_threads = []
try:
    from sqlalchemy import create_engine
    import pandas as pd
    engine = create_engine(PG_DSN)
    df_threads = pd.read_sql("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id LIMIT 200", con=engine)
    existing_threads = df_threads["thread_id"].tolist()
except Exception:
    pass

resume_choice = None
if existing_threads:
    resume_choice = st.sidebar.selectbox("Reanudar hilo existente (opcional)", ["(ninguno)"] + existing_threads, index=0)

# Decide thread_id
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "chat-session-1"

if resume_choice and resume_choice != "(ninguno)":
    st.session_state.thread_id = resume_choice
elif user_name:
    base = slugify(user_name) or f"anon-{uuid.uuid4().hex[:6]}"
    if mode == "Continuar por nombre":
        st.session_state.thread_id = base
    else:  # Crear nuevo
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        st.session_state.thread_id = f"{base}-{ts}"

st.sidebar.write(f"**thread_id actual:** `{st.session_state.thread_id}`")


# ─────────────────────────────# ─────────────────────────────# ─────────────────────────────# ─────────────────────────────
# === Historial de chats (tablas propias, separadas de las de LangGraph) ===
engine_hist = create_engine(PG_DSN)

DDL = """
CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY,
  title TEXT,
  thread_id TEXT UNIQUE,
  user_label TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chat_messages (
  id BIGSERIAL PRIMARY KEY,
  conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
  role TEXT CHECK (role IN ('user','assistant','tool')) NOT NULL,
  content TEXT NOT NULL,
  ts TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_chat_messages_conv_ts ON chat_messages(conversation_id, ts DESC);
"""
with engine_hist.begin() as conn:
    for stmt in DDL.strip().split(";\n\n"):
        if stmt.strip():
            conn.exec_driver_sql(stmt)

def _slugify(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', (s or '').lower()).strip('-')

def create_conversation(user_label: str | None, base_title: str | None = None) -> dict:
    conv_id = str(uuid.uuid4())
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = _slugify(user_label) or "anon"
    thread_id = f"{base}-{ts}-{conv_id[:8]}"
    title = (base_title or "Nueva conversación")[:80]
    with engine_hist.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO conversations(id, title, thread_id, user_label) "
                "VALUES (:id, :title, :thread_id, :user_label)"
            ),
            {"id": conv_id, "title": title, "thread_id": thread_id, "user_label": user_label},
        )
    return {"id": conv_id, "thread_id": thread_id, "title": title}


def upsert_conversation_for_name(user_label: str) -> dict:
    base = _slugify(user_label)
    with engine_hist.begin() as conn:
        row = conn.execute(
            text(
                "SELECT id, title, thread_id "
                "FROM conversations "
                "WHERE user_label = :u "
                "ORDER BY created_at DESC LIMIT 1"
            ),
            {"u": user_label},
        ).mappings().first()
    if row:
        return dict(row)
    return create_conversation(user_label=user_label, base_title=f"Chat de {user_label}")


def list_conversations(query: str | None = None, limit: int = 50) -> pd.DataFrame:
    q = """
    SELECT c.id, c.title, c.thread_id, c.user_label,
           COALESCE(MAX(m.ts), c.created_at) AS last_ts,
           COUNT(m.id) AS msg_count
    FROM conversations c
    LEFT JOIN chat_messages m ON m.conversation_id = c.id
    {where}
    GROUP BY c.id
    ORDER BY last_ts DESC
    LIMIT :limit
    """
    where = ""
    params = {"limit": limit}
    if query:
        where = "WHERE c.title ILIKE :q OR c.user_label ILIKE :q OR c.thread_id ILIKE :q"
        params["q"] = f"%{query}%"
    with engine_hist.begin() as conn:
        return pd.read_sql(text(q.format(where=where)), con=conn, params=params)

def get_messages(conversation_id: str, limit: int = 500) -> pd.DataFrame:
    with engine_hist.begin() as conn:
        return pd.read_sql(
            text("SELECT role, content, ts FROM chat_messages WHERE conversation_id=:cid ORDER BY ts ASC LIMIT :lim"),
            con=conn, params={"cid": conversation_id, "lim": limit}
        )

def log_message(conversation_id: str, role: str, content: str):
    with engine_hist.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO chat_messages(conversation_id, role, content) "
                "VALUES (:cid, :role, :content)"
            ),
            {"cid": conversation_id, "role": role, "content": content},
        )
        conn.execute(
            text("UPDATE conversations SET updated_at = NOW() WHERE id = :cid"),
            {"cid": conversation_id},
        )



def rename_conversation(conversation_id: str, new_title: str):
    with engine_hist.begin() as conn:
        conn.execute(
            text(
                "UPDATE conversations "
                "SET title = :t, updated_at = NOW() "
                "WHERE id = :cid"
            ),
            {"t": new_title[:80], "cid": conversation_id},
        )


def delete_conversation(conversation_id: str):
    with engine_hist.begin() as conn:
        conn.execute(
            text("DELETE FROM conversations WHERE id = :cid"),
            {"cid": conversation_id},
        )

# ───────────────────────────── Sidebar ─────────────────────────────
st.sidebar.header("⚙️ Parámetros")
top_k = st.sidebar.slider("Top-K del retriever", 1, 10, 4, 1)
use_title_extractor = st.sidebar.checkbox("Extraer títulos (usa LLM, gasta tokens)", value=False)
st.sidebar.markdown("### 🗂️ Estrategia de índice ES")
index_strategy = st.sidebar.selectbox("Guardar y consultar documentos por:", ["Global", "Por usuario", "Por hilo"], index=2)
if st.sidebar.button("♻️ Reset caches"):
    st.cache_resource.clear()
    st.cache_data.clear()
    st.success("Caches limpiadas. Vuelve a intentar la operación.")

# Asegura que tengamos user_name y thread_id en session_state
user_name = st.session_state.get("user_name")
thread_id = st.session_state.get("thread_id", "chat-session-1")

# Calcula índice activo y guárdalo
st.session_state.active_es_index = compute_es_index(index_strategy, user_name, thread_id)
st.sidebar.write(f"**Índice activo:** `{st.session_state.active_es_index}`")

# ─── Panel lateral: ver memoria en Postgres (tolerante a esquemas) ────────────
import pandas as pd
from sqlalchemy import create_engine

def table_columns(engine, table):
    q = """
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_schema='public' AND table_name=%s
    ORDER BY ordinal_position
    """
    df = pd.read_sql(q, con=engine, params=(table,))
    return {r["column_name"]: r["data_type"] for _, r in df.iterrows()}

def pick_first(candidates, cols_dict):
    for c in candidates:
        if c in cols_dict:
            return c
    return None

with st.sidebar.expander("🧠 Memoria (Postgres)"):
    try:
        engine = create_engine(PG_DSN)

        # Tablas detectadas
        tables = pd.read_sql(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema='public'
              AND table_name IN ('checkpoints','checkpoint_blobs','writes','checkpoint_writes','checkpoint_migrations')
            ORDER BY table_name
            """,
            con=engine,
        )
        st.write("Tablas:", tables)
        existing = set(tables["table_name"].tolist())

        # ----- CHECKPOINTS -----
        if "checkpoints" in existing:
            ck_cols = table_columns(engine, "checkpoints")

            # timestamp preferido (si existe)
            ts_col = pick_first(["created_at", "ts", "updated_at", "inserted_at", "timestamp"], ck_cols)

            # columnas seguras para mostrar
            base_cols = [c for c in ["thread_id", "checkpoint_ns", "checkpoint_id", "parent_checkpoint_id"] if c in ck_cols]
            sel_cols = base_cols + ([ts_col] if ts_col else [])
            order_clause = f"ORDER BY {ts_col} DESC" if ts_col else ("ORDER BY checkpoint_id DESC" if "checkpoint_id" in ck_cols else "")

            if sel_cols:
                sql = f"SELECT {', '.join(sel_cols)} FROM checkpoints {order_clause} LIMIT 20"
            else:
                sql = "SELECT * FROM checkpoints LIMIT 20"  # ultra fallback

            df_ck = pd.read_sql(sql, con=engine)
            st.write("Últimos checkpoints:", df_ck)

            # Threads únicos
            if "thread_id" in ck_cols:
                df_threads = pd.read_sql("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id", con=engine)
                st.write("Threads:", df_threads)
        else:
            st.info("Tabla 'checkpoints' no encontrada.")

        # ----- WRITES / CHECKPOINT_WRITES -----
        writes_table = "checkpoint_writes" if "checkpoint_writes" in existing else ("writes" if "writes" in existing else None)
        if writes_table:
            w_cols = table_columns(engine, writes_table)
            tsw = pick_first(["created_at", "ts", "updated_at", "inserted_at", "timestamp"], w_cols)
            base_w = [c for c in ["thread_id", "checkpoint_id"] if c in w_cols]
            sel_w = base_w + ([tsw] if tsw else [])
            order_w = f"ORDER BY {tsw} DESC" if tsw else ("ORDER BY checkpoint_id DESC" if "checkpoint_id" in w_cols else "")
            sql_w = f"SELECT {', '.join(sel_w) if sel_w else '*'} FROM {writes_table} {order_w} LIMIT 20"
            df_w = pd.read_sql(sql_w, con=engine)
            st.write(f"Últimos writes ({writes_table}):", df_w)
        else:
            st.info("Tabla de writes no encontrada ('writes' o 'checkpoint_writes').")

        # ----- CHECKPOINT_BLOBS -----
        if "checkpoint_blobs" in existing:
            b_cols = table_columns(engine, "checkpoint_blobs")
            tsb = pick_first(["created_at", "ts", "updated_at", "inserted_at", "timestamp"], b_cols)
            size_col = pick_first(["value", "blob", "state", "data"], b_cols)  # nombre cambia entre versiones

            order_b = f"ORDER BY {tsb} DESC" if tsb else ("ORDER BY checkpoint_id DESC" if "checkpoint_id" in b_cols else "")
            if size_col:
                # Si no es bytea, medimos tamaño aproximado convirtiéndolo a texto (para jsonb, etc.)
                is_bytea = (b_cols[size_col] == "bytea")
                size_expr = f"OCTET_LENGTH({size_col})" if is_bytea else f"OCTET_LENGTH(convert_to({size_col}::text,'UTF8'))"
                base_b = [c for c in ["thread_id", "checkpoint_id"] if c in b_cols]
                sel_b = base_b + ([tsb] if tsb else []) + [f"{size_expr} AS bytes"]
                sql_b = f"SELECT {', '.join(sel_b)} FROM checkpoint_blobs {order_b} LIMIT 20"
            else:
                # si no encontramos columna de contenido, muestra metadatos
                base_b = [c for c in ["thread_id", "checkpoint_id"] if c in b_cols]
                sel_b = base_b + ([tsb] if tsb else [])
                sql_b = f"SELECT {', '.join(sel_b) if sel_b else '*'} FROM checkpoint_blobs {order_b} LIMIT 20"

            df_blobs = pd.read_sql(sql_b, con=engine)
            st.write("Blobs (vista rápida):", df_blobs)
        else:
            st.info("Tabla 'checkpoint_blobs' no encontrada.")

        # ----- MIGRACIONES -----
        if "checkpoint_migrations" in existing:
            try:
                df_mig = pd.read_sql(
                    "SELECT * FROM checkpoint_migrations ORDER BY applied_at DESC NULLS LAST LIMIT 5",
                    con=engine,
                )
            except Exception:
                # fallback si no existe 'applied_at'
                df_mig = pd.read_sql("SELECT * FROM checkpoint_migrations LIMIT 5", con=engine)
            st.write("Migraciones:", df_mig)

    except Exception as e:
        st.error(f"❌ No pude leer las tablas de memoria: {e}")

st.sidebar.markdown("### 🗂️ Historial de chats")



# Nombre/ID de usuario para asociar hilos
user_label = st.sidebar.text_input("Tu nombre o ID", value=st.session_state.get("user_name",""))
if user_label:
    st.session_state.user_name = user_label

# Buscar conversaciones
q = st.sidebar.text_input("Buscar en historial (título/usuario/thread_id)", "")

# Listado
df_convs = list_conversations(query=q, limit=200)
if df_convs.empty:
    st.sidebar.info("No hay conversaciones aún.")
else:
    for _, row in df_convs.iterrows():
        with st.sidebar.expander(f"🗨️ {row['title']}  ·  {row['msg_count']} msg  ·  {row['last_ts']}"):
            st.write(f"ID: `{row['id']}`")
            st.write(f"thread_id: `{row['thread_id']}`")
            st.write(f"usuario: `{row.get('user_label','')}`")

            colA, colB, colC = st.columns([1,1,1])
            with colA:
                if st.button("Abrir", key=f"open_{row['id']}"):
                    st.session_state.conversation_id = row["id"]
                    st.session_state.thread_id = row["thread_id"]
                    st.session_state.chat_history = get_messages(row["id"]).to_dict("records")
                    st.rerun()
            with colB:
                new_title = st.text_input("Renombrar", value=row["title"], key=f"rename_{row['id']}")
                if st.button("Guardar", key=f"save_{row['id']}"):
                    rename_conversation(row["id"], new_title)
                    st.rerun()
            with colC:
                if st.button("🗑️ Borrar", key=f"del_{row['id']}"):
                    delete_conversation(row["id"])
                    st.rerun()

# Crear conversación nueva
if st.sidebar.button("➕ Nueva conversación"):
    base_title = "Nueva conversación"
    if user_label:
        conv = create_conversation(user_label=user_label, base_title=base_title)
    else:
        conv = create_conversation(user_label=None, base_title=base_title)
    st.session_state.conversation_id = conv["id"]
    st.session_state.thread_id = conv["thread_id"]
    st.session_state.chat_history = []
    st.rerun()

# ───────────────────── ES & Embeddings (LlamaIndex) ─────────────────────

from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core import VectorStoreIndex, StorageContext, Settings
import streamlit as st

@st.cache_resource
def get_vector_store_cached(es_url, es_user, es_password, index_name):
    es_client = ESClient(
        es_url,
        basic_auth=(es_user, es_password) if (es_user or es_password) else None,
        request_timeout=60,
    )
    vs = ElasticsearchStore(
        es_client=es_client,        # ← cliente síncrono
        index_name=index_name,
        use_async=False,            # ← forzar ruta síncrona
        create_index_if_not_exists=True,
        text_field="content",
        vector_field="embedding",
        #metadata_field="metadata",
        #id_field="id",
    )
    # “Cinturón y tirantes”: asegura que el store interno sea sync
    try:
        if hasattr(vs, "_use_async"):
            vs._use_async = False
        if hasattr(vs, "_sync_store") and hasattr(vs, "_store"):
            vs._store = vs._sync_store
    except Exception:
        pass
    return vs


def get_vector_store(index_name: str | None = None) -> ElasticsearchStore:
    index_name = index_name or st.session_state.get("active_es_index") or ES_INDEX_BASE
    return get_vector_store_cached(ES_URL, ES_USER, ES_PASSWORD, index_name)
def get_index(index_name: str | None = None) -> VectorStoreIndex:
    ensure_embed_model()
    vs = get_vector_store(index_name)
    key = f"vector_index::{vs.index_name}"
    if key not in st.session_state:
        sc = StorageContext.from_defaults(vector_store=vs)
        st.session_state[key] = VectorStoreIndex.from_vector_store(
            vector_store=vs,
            storage_context=sc,
            embed_model=Settings.embed_model,
        )
    return st.session_state[key]



def get_retriever():
    idx = get_index(st.session_state.get("active_es_index"))
    return idx.as_retriever(similarity_top_k=top_k)





def ensure_embed_model():
    """Embeddings HF (BGE small). Debe coincidir con lo usado al indexar."""
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        device=DEVICE,
    )

def ensure_llm_for_extractors():
    """LLM para transformaciones que lo requieran (TitleExtractor, etc.)."""
    Settings.llm = LI_OpenAI(model="gpt-4o-mini", temperature=0.2)

from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.elasticsearch import ElasticsearchStore as LIElasticsearchStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core import Document  # si recreas documentos limpios

def ingest_files_into_es(filepaths):
    ensure_embed_model()  # tu función: HuggingFaceEmbedding("BAAI/bge-small-en-v1.5", device=DEVICE)
    if use_title_extractor:
        ensure_llm_for_extractors()

    # Carga y limpieza
    parent_dir = str(filepaths[0].parent)
    docs = SimpleDirectoryReader(parent_dir).load_data()
    clean_docs = []
    for d in docs:
        txt = " ".join(d.text.split())
        clean_docs.append(Document(text=txt, metadata=d.metadata, id_=getattr(d, "id_", None)))

    # 🔹 índice por hilo/usuario si quieres aislar uploads
    idxname = f"{ES_INDEX}--{st.session_state.thread_id.replace(' ', '-')}".lower()

    vector_store = LIElasticsearchStore(
        es_url=ES_URL,
        es_user=ES_USER,
        es_password=ES_PASSWORD,
        index_name=idxname,
        # (opcional) nombres de campos si quieres personalizarlos
        # text_field="content", vector_field="embedding"
    )
    sc = StorageContext.from_defaults(vector_store=vector_store)

    # Construye e inserta
    VectorStoreIndex.from_documents(clean_docs, storage_context=sc)

    return len(clean_docs), [p.name for p in filepaths], idxname



# def get_retriever():
#     """Crea retriever desde ES con embeddings configurados."""
#     ensure_embed_model()
#     vector_store = get_vector_store()
#     storage_context = StorageContext.from_defaults(vector_store=vector_store)
#     index = VectorStoreIndex.from_vector_store(
#         vector_store=vector_store,
#         storage_context=storage_context,
#         embed_model=Settings.embed_model,
#     )
#     return index.as_retriever(similarity_top_k=top_k)

# ───────────────────────── Tool (RAG) ─────────────────────────
from llama_index.core.postprocessor import SimilarityPostprocessor
from langchain.tools import tool

@tool
def consulta_corpus(query: str) -> str:
    """Busca respuestas en el índice activo (según estrategia elegida)."""
    retriever = get_retriever()
    nodes = retriever.retrieve(query)
    nodes = SimilarityPostprocessor(similarity_cutoff=0.75).postprocess_nodes(nodes)
    if not nodes:
        return "Sin evidencia suficiente en el corpus para esta pregunta."
    blocks = []
    for i, n in enumerate(nodes, 1):
        meta = n.node.metadata or {}
        src = meta.get("source") or meta.get("file_name") or "N/A"
        page = meta.get("page_label") or meta.get("page") or ""
        src_str = f"{src}{f', p. {page}' if page else ''}"
        blocks.append(f"[{i}] {n.node.get_content().strip()}\nFuente: {src_str}")
    return "\n\n".join(blocks)


# ───────────────────────── Memoria (Postgres) ─────────────────────────
if "pg_pool" not in st.session_state:
    st.session_state.pg_pool = ConnectionPool(
        conninfo=PG_DSN,
        max_size=20,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    )
if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = PostgresSaver(st.session_state.pg_pool)
    st.session_state.checkpointer.setup()  # crea tablas si faltan

# ───────────────────────── LLM + Agente ReAct ─────────────────────────
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)

system_prompt = """EEres un asistente que puede usar herramientas SOLO cuando el usuario pida información del corpus/documentos.
Reglas:
- Si el mensaje es saludo, cortesía, charla pequeña o no pide datos del corpus → RESPONDE tú mismo, NO uses herramientas.
- Usa la herramienta `consulta_corpus` solo si la intención es buscar contenido en el corpus (documentos subidos / índice).
- Si no hay evidencia suficiente en el corpus, dilo y sugiere subir/ingestar más fuentes.
- Responde en español y cita la fuente (archivo y página) cuando uses el corpus.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{messages}"),
    ]
)

toolkit = [consulta_corpus]
agent = create_react_agent(
    llm,
    toolkit,
    checkpointer=st.session_state.checkpointer,
    prompt=prompt,
)


# ───────────────────────── UI: Subida de archivos ─────────────────────────
st.subheader("📎 Subir archivos al corpus")
uploaded_files = st.file_uploader(
    "Arrastra aquí PDF/TXT/DOCX/MD (se indexarán en Elasticsearch)",
    accept_multiple_files=True,
    type=["pdf", "txt", "docx", "md"],
)

if uploaded_files:
    with st.spinner("🔄 Ingestando archivos..."):
        tmpdir = Path(tempfile.mkdtemp(prefix="rag_uploads_"))
        paths = []
        for uf in uploaded_files:
            dest = tmpdir / uf.name
            with open(dest, "wb") as f:
                f.write(uf.read())
            paths.append(dest)

        count, names, idxname = ingest_files_into_es(paths)
        st.success(f"✅ Ingestados {count} doc(s) en índice **{idxname}**: {', '.join(names)}")


# ───────────────────────── UI: Chat ─────────────────────────
# Conversación activa por defecto
if "conversation_id" not in st.session_state:
    # Si hay nombre, intenta reusar; si no, crea una nueva
    if st.session_state.get("user_name"):
        conv = upsert_conversation_for_name(st.session_state["user_name"])
    else:
        conv = create_conversation(user_label=None, base_title="Nueva conversación")
    st.session_state.conversation_id = conv["id"]
    st.session_state.thread_id = conv["thread_id"]
    st.session_state.chat_history = get_messages(conv["id"]).to_dict("records")

st.subheader(f"💬 {st.session_state.get('user_name','(anónimo)')} — hilo `{st.session_state.thread_id}`")
st.write(f"Vector store ES → índice activo: `{st.session_state.get('active_es_index')}`")

# Render histórico desde DB
for m in st.session_state.chat_history:
    st.chat_message(m["role"]).write(m["content"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "chat-session-1"

st.subheader("💬 Chat")
# render previo
for m in st.session_state.chat_history:
    st.chat_message(m["role"]).write(m["content"])

user_query = st.chat_input("Escribe tu pregunta…")
if user_query:
    # Guarda mensaje del usuario
    log_message(st.session_state.conversation_id, "user", user_query)
    st.chat_message("user").write(user_query)

    # Pregunta al agente (misma memoria: thread_id)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=user_query)]},
                config={"configurable": {"thread_id": st.session_state.thread_id}},
            )
            final_text = result["messages"][-1].content if result.get("messages") else "⚠️ Sin respuesta."
        except Exception as e:
            final_text = f"⚠️ Error durante la generación: {e}"

        # Guarda respuesta del asistente y muestra
        log_message(st.session_state.conversation_id, "assistant", final_text)
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({"role": "assistant", "content": final_text})
        placeholder.write(final_text)


# ───────────────────────── Sidebar: Estado ─────────────────────────
with st.sidebar:
    st.markdown("### Estado")
    st.write(f"Vector store: **Elasticsearch** → índice: `{ES_INDEX}`")
    st.write(f"Embeddings HF: **BAAI/bge-small-en-v1.5** ({'GPU' if DEVICE=='cuda' else 'CPU'})")
    st.write(f"Top-K: **{top_k}**")
    st.write("Memoria: **PostgresSaver**")
    if LG_API:
        st.success("LangSmith tracing: ON")
    else:
        st.info("LangSmith tracing: OFF (agrega langgraphapi.txt para activar)")
with st.sidebar.expander("📤 Exportar conversación activa"):
    if "conversation_id" in st.session_state:
        msgs_df = get_messages(st.session_state.conversation_id)
        records = msgs_df.to_dict("records")

        def _json_default(o):
            if isinstance(o, (pd.Timestamp, dt.datetime, dt.date)):
                # ISO 8601
                try:
                    return o.isoformat()
                except Exception:
                    return str(o)
            return str(o)

        payload = json.dumps(records, ensure_ascii=False, indent=2, default=_json_default).encode("utf-8")
        st.download_button(
            "Descargar JSON",
            data=payload,
            file_name=f"conv_{st.session_state.conversation_id}.json",
            mime="application/json",
        )

