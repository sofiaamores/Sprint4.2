from pydantic import BaseModel, Field, field_validator
from typing import List


class LinkedinPost(BaseModel):
    title: str = Field(
        ...,
        min_length=5,
        max_length=120,
        description="Título atractivo del post de LinkedIn"
    )

    content: str = Field(
        ...,
        min_length=20,
        max_length=3000,
        description="Contenido principal del post"
    )

    hashtags: List[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Lista de hashtags relevantes"
    )

    category: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="Categoría del post (ej: tecnología, liderazgo, marketing)"
    )

    @field_validator("hashtags")
    @classmethod
    def validate_hashtags(cls, v: List[str]):
        for tag in v:
            if not tag.startswith("#"):
                raise ValueError("Cada hashtag debe comenzar con '#'")
        return v
