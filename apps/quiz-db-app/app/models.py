"""
models.py – Pydantic schemas for true/false healthcare questions.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from enum import Enum


class Category(str, Enum):
    anatomy = "Anatomy"
    pharmacology = "Pharmacology"
    pathology = "Pathology"
    nursing = "Nursing"
    nutrition = "Nutrition"
    public_health = "Public Health"
    first_aid = "First Aid"
    mental_health = "Mental Health"
    general = "General"


class QuestionCreate(BaseModel):
    text: str = Field(..., min_length=5, max_length=500, description="Question text")
    correct_answer: bool = Field(..., description="True or False")
    category: Category = Category.general
    explanation: str = Field(
        default="", max_length=1000, description="Why the answer is correct"
    )
    difficulty: int = Field(default=1, ge=1, le=3, description="1=easy 2=medium 3=hard")


class QuestionUpdate(BaseModel):
    text: str | None = None
    correct_answer: bool | None = None
    category: Category | None = None
    explanation: str | None = None
    difficulty: int | None = Field(default=None, ge=1, le=3)


class QuestionOut(BaseModel):
    id: str
    text: str
    correct_answer: bool
    category: str
    explanation: str
    difficulty: int
