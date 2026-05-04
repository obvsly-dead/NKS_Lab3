# api_server.py
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from model_utils import generate_answer

app = FastAPI(title="NKS Lab3 Qwen Chat API", version="1.0")
app.mount("/static", StaticFiles(directory="static"), name="static")


class ChatMessage(BaseModel):
    user: str
    bot: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    response: str
    latency: float
    tokens: int
    throughput: float


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        history = [item.model_dump() for item in request.history]
        result = generate_answer(request.message, history)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка инференса: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
