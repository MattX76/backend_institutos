from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()

POSTGRES_URI = os.getenv("POSTGRES_URI")
if not POSTGRES_URI:
    raise ValueError("POSTGRES_URI no está definida en el archivo .env")

engine = create_engine(POSTGRES_URI)

DDL = """
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    file_name TEXT,
    status VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_tenant_id ON documents (tenant_id);
"""

def setup_database():
    try:
        with engine.connect() as connection:
            print("Conectado a PostgreSQL. Creando tablas si no existen...")
            connection.execute(text(DDL))
            connection.commit()
        print("✅ Tablas verificadas/creadas exitosamente.")
    except Exception as e:
        print(f"❌ Error al conectar o configurar la base de datos: {e}")

if __name__ == "__main__":
    setup_database()