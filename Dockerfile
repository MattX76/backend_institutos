# Usa una imagen oficial de Python
FROM python:3.11-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /code

# Copia solo el archivo de requerimientos primero para aprovechar la caché de Docker
COPY ./requirements.txt /code/requirements.txt

# Instala las dependencias de Python (incluyendo la versión CPU de torch)
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copia el resto de la aplicación (la carpeta api)
COPY ./api /code/api

# Expone el puerto que Uvicorn usará
EXPOSE 8000

# Comando para correr la aplicación
# Usamos --host 0.0.0.0 para que sea accesible desde fuera del contenedor
CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "8000"]