# main.py
from fastapi import FastAPI

app = FastAPI(
    title="API de Inferencia",
    description="API educativa con FastAPI",
    version="1.0.0"
)

@app.get("/")
def hola_mundo():
    return {"mensaje": "Hola Mundo de Jackson"}


@app.get("/health")
def health_check():
    return {"status": "ok"}
