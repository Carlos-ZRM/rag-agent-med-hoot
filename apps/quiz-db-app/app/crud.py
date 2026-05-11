"""
crud.py – Backend business logic for question CRUD.

Every function receives data as Pydantic models and delegates
persistence to the abstract DBEngine from db.py.
"""

from __future__ import annotations

import re
from app.db import get_engine
from app.models import QuestionCreate, QuestionUpdate, QuestionOut

COLLECTION = "questions"


async def create_question(data: QuestionCreate) -> QuestionOut:
    engine = get_engine()
    doc = data.model_dump()
    doc["category"] = doc["category"].value if hasattr(doc["category"], "value") else doc["category"]
    doc_id = await engine.insert(COLLECTION, doc)
    return QuestionOut(id=doc_id, **doc)


async def get_question(question_id: str) -> QuestionOut | None:
    engine = get_engine()
    doc = await engine.find_one(COLLECTION, {"id": question_id})
    if doc is None:
        return None
    return QuestionOut(**doc)


async def list_questions(
    skip: int = 0,
    limit: int = 50,
    search: str = "",
    category: str = "",
    difficulty: int | None = None,
) -> list[QuestionOut]:
    engine = get_engine()
    filt: dict = {}

    if search:
        # MongoDB uses $regex; Couchbase engine translates $regex → LIKE
        filt["text"] = {"$regex": re.escape(search), "$options": "i"}

    if category:
        filt["category"] = category

    if difficulty is not None:
        filt["difficulty"] = difficulty

    docs = await engine.find_many(COLLECTION, filt, skip=skip, limit=limit)
    return [QuestionOut(**d) for d in docs]


async def count_questions(
    search: str = "",
    category: str = "",
    difficulty: int | None = None,
) -> int:
    engine = get_engine()
    filt: dict = {}
    if search:
        filt["text"] = {"$regex": re.escape(search), "$options": "i"}
    if category:
        filt["category"] = category
    if difficulty is not None:
        filt["difficulty"] = difficulty
    return await engine.count(COLLECTION, filt)


async def update_question(question_id: str, data: QuestionUpdate) -> QuestionOut | None:
    engine = get_engine()
    fields = data.model_dump(exclude_none=True)
    if "category" in fields and hasattr(fields["category"], "value"):
        fields["category"] = fields["category"].value
    if not fields:
        return await get_question(question_id)
    ok = await engine.update(COLLECTION, question_id, fields)
    if not ok:
        return None
    return await get_question(question_id)


async def delete_question(question_id: str) -> bool:
    engine = get_engine()
    return await engine.delete(COLLECTION, question_id)
