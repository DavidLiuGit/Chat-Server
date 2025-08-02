from typing import Literal
from pydantic import BaseModel


class Model(BaseModel):
    """
    See OpenAI API reference for Retrieve Model
    https://platform.openai.com/docs/api-reference/models/retrieve
    https://platform.openai.com/docs/api-reference/models/object
    """
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str


class ListModelsResponse(BaseModel):
    """
    See OpenAI API reference for List Models
    https://platform.openai.com/docs/api-reference/models/list
    """
    object: Literal["list"] = "list"
    data: list[Model]
