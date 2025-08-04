from fastapi import APIRouter

from app.models.v1.models import Model, ListModelsResponse

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
    response_model=ListModelsResponse,
    response_model_exclude_none=True,
    tags=["Models"],
)
def list_models():
    """
    Return list of available models.
    Reference: https://platform.openai.com/docs/api-reference/models/list
    """
    return ListModelsResponse(data=list(AVAILABLE_MODELS.values()))


@models_router.get(
    "/models/{model}",
    response_model=Model,
    response_model_exclude_none=True,
    tags=["Models"],
)
def retrieve_model(model: str):
    """
    Return metadata on the requested model.
    Reference: https://platform.openai.com/docs/api-reference/models/retrieve
    """
    if model not in AVAILABLE_MODELS:
        return None
    return AVAILABLE_MODELS[model]
