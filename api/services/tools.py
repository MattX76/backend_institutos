# api/services/tools.py

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from api.core.config import settings
from api.services.query import query_index # Reutilizamos nuestra función RAG síncrona

# Define el LLM que usarán las herramientas para el razonamiento interno
llm = ChatOpenAI(model=settings.LLM_MODEL)

@tool
def simple_rag_query(query: str, tenant_id: str) -> str:
    """Útil para responder preguntas directas y específicas sobre el contenido de la normativa.
    Usa esta herramienta para consultas simples."""
    print(f"-> Ejecutando RAG simple para '{query}' en tenant '{tenant_id}'")
    response = query_index(query, tenant_id)
    
    # Formatear la respuesta para incluir contexto y fuentes
    context_str = "\n\n".join([f"Fuente: {s.metadata.get('file_name', 'N/A')}\n{s.text}" for s in response.sources])
    return f"Respuesta: {response.answer}\n\nContexto Utilizado:\n{context_str}"


@tool
def compliance_checklist_generator(topic: str, tenant_id: str) -> str:
    """Útil para generar un checklist de cumplimiento o una lista de requisitos detallada 
    sobre un tema normativo específico, como 'crear una nueva carrera' o 'requisitos de infraestructura'."""
    print(f"-> Iniciando generación de checklist para '{topic}' en tenant '{tenant_id}'")

    # 1. Descomposición (sin cambios)
    decomposition_prompt = (
        f"Basado en el tema '{topic}', genera una lista de 3 a 5 sub-preguntas "
        f"clave para investigar en la normativa de creación de IES. "
        f"Devuelve solo la lista de preguntas, separadas por saltos de línea."
    )
    response = llm.invoke(decomposition_prompt)
    sub_questions = [q for q in response.content.strip().split("\n") if q]
    print(f"   Sub-preguntas generadas: {sub_questions}")

    # 2. Búsqueda (sin cambios)
    evidence_context = ""
    for q in sub_questions:
        rag_response = query_index(q, tenant_id)
        for source in rag_response.sources:
            evidence_context += source.text + "\n"
    
    evidence_context = evidence_context.strip()
    print(f"   Evidencia recopilada (longitud: {len(evidence_context)})")

    # --- INICIO DE LA CORRECCIÓN ---

    # 3. Verificación: Si no hay evidencia, no continuamos.
    if not evidence_context:
        print("   No se encontró evidencia. Devolviendo respuesta predeterminada.")
        return "No he encontrado información relevante en la base de conocimiento para responder a esta consulta."

    # 4. Síntesis: Prompt mucho más estricto
    synthesis_prompt = (
        f"Actúa como un consultor experto en normativa de IES. Basándote ÚNICA Y EXCLUSIVAMENTE en la "
        f"siguiente evidencia extraída de la normativa, genera un checklist de cumplimiento "
        f"detallado para el tema '{topic}'.\n"
        f"Si la evidencia proporcionada no es suficiente para crear un checklist, responde EXACTAMENTE: "
        f"'La información encontrada en la base de conocimiento no es suficiente para generar el checklist solicitado.'\n"
        f"NO utilices tu conocimiento general. Cita la fuente o idea principal para cada punto.\n\n"
        f"--- EVIDENCIA RECOPILADA ---\n{evidence_context}\n\n"
        f"--- CHECKLIST DE CUMPLIMIENTO PARA '{topic}' ---"
    )
    
    final_response = llm.invoke(synthesis_prompt)
    print("-> Checklist generado exitosamente.")
    return final_response.content
# Agrupamos las herramientas para el agente
langchain_tools = [simple_rag_query, compliance_checklist_generator]