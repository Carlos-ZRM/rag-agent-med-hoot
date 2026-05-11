"""
main.py – FastAPI application with Uvicorn entry point.

Routes:
  WEB UI
    GET  /                  → dashboard / question manager
  REST API
    GET    /api/questions    → list + search + filter
    POST   /api/questions    → create
    GET    /api/questions/{id}
    PUT    /api/questions/{id}
    DELETE /api/questions/{id}
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.db import get_engine
from app.models import QuestionCreate, QuestionUpdate, Category
from app import crud

logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Lifespan – connect / disconnect DB
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = get_engine()
    await engine.connect()
    yield
    await engine.disconnect()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Health Kahoot – Question Manager",
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ---------------------------------------------------------------------------
# Web UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    categories = [c.value for c in Category]

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"categories": categories},
    )


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------

@app.get("/api/questions")
async def api_list(
    search: str = Query("", max_length=200),
    category: str = Query(""),
    difficulty: int | None = Query(None, ge=1, le=3),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    items = await crud.list_questions(skip, limit, search, category, difficulty)
    total = await crud.count_questions(search, category, difficulty)
    return {"items": [q.model_dump() for q in items], "total": total}


@app.post("/api/questions", status_code=201)
async def api_create(data: QuestionCreate):
    q = await crud.create_question(data)
    return q.model_dump()


@app.get("/api/questions/{qid}")
async def api_get(qid: str):
    q = await crud.get_question(qid)
    if q is None:
        raise HTTPException(404, "Question not found")
    return q.model_dump()


@app.put("/api/questions/{qid}")
async def api_update(qid: str, data: QuestionUpdate):
    q = await crud.update_question(qid, data)
    if q is None:
        raise HTTPException(404, "Question not found")
    return q.model_dump()


@app.delete("/api/questions/{qid}")
async def api_delete(qid: str):
    ok = await crud.delete_question(qid)
    if not ok:
        raise HTTPException(404, "Question not found")
    return JSONResponse({"deleted": True})


# ---------------------------------------------------------------------------
# Uvicorn entry
# ---------------------------------------------------------------------------

def run():
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run()
