from fastapi import APIRouter
from openai.types import Model
from openai.pagination import SyncPage

AVAILABLE_MODELS: dict[str, Model] = {
    # example:
    "custom_model": Model(
        id="custom_model",
        object="model",
        created=1677610602,
        owned_by="organization-owner",
    )
}
"""
TODO: replace with metadata for your own custom model.
Maps model ID to `Model` object.
"""

models_router = APIRouter()


@models_router.get(
    "/models",
    response_model_exclude_none=True,
    tags=["Models"],
)
def list_models() -> SyncPage[Model]:
    """
    Return list of available models.
    Reference: https://platform.openai.com/docs/api-reference/models/list
    """
    return SyncPage(data=list(AVAILABLE_MODELS.values()), object="list")


@models_router.get(
    "/models/{model}",
    response_model_exclude_none=True,
    tags=["Models"],
)
def retrieve_model(model: str) -> Model | None:
    """
    Return metadata on the requested model.
    Reference: https://platform.openai.com/docs/api-reference/models/retrieve
    """
    return AVAILABLE_MODELS.get(model)
