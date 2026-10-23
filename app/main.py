from fastapi import FastAPI


app = FastAPI(
    title="Sampada AI",
    description=(
        "A UNESCO World Heritage chatbot using"
        "metadata filtering and RAG"
    ),
    version="0.1.0"
)


@app.get("/")
def root():
    return {
        "message": "Welcome to Sampada AI",
        "status": "running"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}