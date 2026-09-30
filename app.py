"""FastAPI backend and static web app for ChatGPT Memory Search."""

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.rag import answer_question
from src.retriever import Retriever

ROOT = Path(__file__).resolve().parent
FRONTEND_INDEX = ROOT / "frontend" / "index.html"
retriever = None
load_dotenv()
DEMO_USERNAME = os.getenv("DEMO_USERNAME")
DEMO_PASSWORD = os.getenv("DEMO_PASSWORD")
SESSION_SECRET = os.getenv("SESSION_SECRET")
if not SESSION_SECRET:
    raise RuntimeError("SESSION_SECRET must be set in .env or the deployment environment.")
SESSION_MAX_AGE = 14 * 24 * 60 * 60


def make_session_cookie(username: str) -> str:
    payload = json.dumps(
        {"username": username, "expires": int(time.time()) + SESSION_MAX_AGE},
        separators=(",", ":"),
    ).encode("utf-8")
    encoded = base64.urlsafe_b64encode(payload).rstrip(b"=")
    signature = hmac.new(SESSION_SECRET.encode("utf-8"), encoded, hashlib.sha256).digest()
    return encoded.decode("ascii") + "." + base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")


def session_username(request: Request) -> str | None:
    token = request.cookies.get("rag_session", "")
    try:
        encoded, supplied_signature = token.split(".", 1)
        expected_signature = base64.urlsafe_b64encode(
            hmac.new(
                SESSION_SECRET.encode("utf-8"),
                encoded.encode("ascii"),
                hashlib.sha256,
            ).digest()
        ).rstrip(b"=").decode("ascii")
        if not hmac.compare_digest(supplied_signature, expected_signature):
            return None
        payload = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        session_data = json.loads(payload)
        if session_data.get("expires", 0) < time.time():
            return None
        username = session_data.get("username")
        return username if isinstance(username, str) and username else None
    except (ValueError, TypeError, UnicodeError, json.JSONDecodeError):
        return None


def require_login(request: Request) -> str:
    username = session_username(request)
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Please log in to continue.",
        )
    return username


class AskRequest(BaseModel):
    question: str


class LoginRequest(BaseModel):
    username: str
    password: str


@asynccontextmanager
async def lifespan(app):
    global retriever
    retriever = Retriever(backend="chroma")
    yield
    retriever = None


app = FastAPI(title="ChatGPT Memory Search", lifespan=lifespan)


@app.get("/")
def frontend():
    return FileResponse(FRONTEND_INDEX)


@app.get("/api/session")
def session_status(request: Request):
    return {"authenticated": bool(session_username(request))}


@app.post("/api/login")
def login(payload: LoginRequest, response: Response):
    username_ok = secrets.compare_digest(payload.username, DEMO_USERNAME or "")
    password_ok = secrets.compare_digest(payload.password, DEMO_PASSWORD or "")
    if not (DEMO_USERNAME and DEMO_PASSWORD and username_ok and password_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="That username or password is incorrect.",
        )
    response.set_cookie(
        "rag_session",
        make_session_cookie(payload.username),
        max_age=SESSION_MAX_AGE,
        httponly=True,
        secure=os.getenv("RENDER", "").lower() == "true",
        samesite="lax",
        path="/",
    )
    return {"authenticated": True}


@app.post("/api/logout")
def logout(response: Response, _username: str = Depends(require_login)):
    response.delete_cookie("rag_session", path="/")
    return {"authenticated": False}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/ask")
def ask(payload: AskRequest, _username: str = Depends(require_login)):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    try:
        result = answer_question(payload.question, retriever)
        if result.get("error"):
            raise RuntimeError(result["error"])
        return {
            "answer": result["answer"],
            "sources": [
                {"title": source["title"], "create_date": source["create_date"], "score": source["score"]}
                for source in result["sources"]
            ],
            "timing": result["timing"],
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Unable to answer question: {error}") from error


if __name__ == "__main__":
    # Run with: uvicorn app:app --reload --port 8000
    # Then open http://localhost:8000 directly; no separate frontend server is needed.
    print("Run with: uvicorn app:app --reload --port 8000")
