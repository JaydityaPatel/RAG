"""FastAPI backend and static web app for ChatGPT Memory Search."""

import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from src.rag import answer_question
from src.retriever import Retriever

ROOT = Path(__file__).resolve().parent
FRONTEND_INDEX = ROOT / "frontend" / "index.html"
retriever = None
load_dotenv()
DEMO_USERNAME = os.getenv("DEMO_USERNAME")
DEMO_PASSWORD = os.getenv("DEMO_PASSWORD")
basic_auth = HTTPBasic(auto_error=False)


def verify_credentials(
    credentials: HTTPBasicCredentials | None = Depends(basic_auth),
) -> str:
    username = credentials.username if credentials else ""
    password = credentials.password if credentials else ""
    # Constant-time comparisons make response timing less useful for guessing
    # credentials one character at a time.
    username_ok = secrets.compare_digest(username, DEMO_USERNAME or "")
    password_ok = secrets.compare_digest(password, DEMO_PASSWORD or "")
    if not (DEMO_USERNAME and DEMO_PASSWORD and username_ok and password_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return username


class AskRequest(BaseModel):
    question: str


@asynccontextmanager
async def lifespan(app):
    global retriever
    retriever = Retriever(backend="chroma")
    yield
    retriever = None


app = FastAPI(title="ChatGPT Memory Search", lifespan=lifespan)


@app.get("/", dependencies=[Depends(verify_credentials)])
def frontend():
    return FileResponse(FRONTEND_INDEX)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/ask", dependencies=[Depends(verify_credentials)])
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    try:
        result = answer_question(request.question, retriever)
        if result.get("error"):
            raise RuntimeError(result["error"])
        return {
            "answer": result["answer"],
            "sources": [
                {"title": source["title"], "create_date": source["create_date"], "score": source["score"]}
                for source in result["sources"]
            ],
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Unable to answer question: {error}") from error


if __name__ == "__main__":
    # Run with: uvicorn app:app --reload --port 8000
    # Then open http://localhost:8000 directly; no separate frontend server is needed.
    print("Run with: uvicorn app:app --reload --port 8000")
