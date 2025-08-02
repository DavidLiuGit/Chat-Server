import os

from boto3 import Session
from botocore.config import Config
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from langchain_aws import ChatBedrock
from langchain_core.language_models import BaseLanguageModel


from logging import getLogger

logger = getLogger(__name__)


def get_language_model() -> BaseLanguageModel:
    """
    Get an instance of BaseLanguageModel, to be used as part of the chain.
    Modify model parameters here, as needed.
    """
    bedrock_client = _get_bedrock_client()
    return ChatBedrock(
        client=bedrock_client,
        # cross-region inference enabled:
        model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        model_kwargs={
            "temperature": 0.69,
        },
        streaming=True,
    )


def get_fast_language_model() -> BaseLanguageModel:
    """
    Get an instance of BaseLanguageModel that is typically faster to complete. Ideal for usage
    as part of a Contextualizer, which in turn is part of a Retriever, or anywhere that latency
    is prioritized over outright model performance.
    """
    bedrock_client = _get_bedrock_client()
    return ChatBedrock(
        client=bedrock_client,
        # cross-region inference enabled
        # use Haiku 3.5, a small but capable model
        model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
        model_kwargs={
            "temperature": 0.69,
        },
        streaming=True,
    )


def _get_bedrock_client() -> BedrockRuntimeClient:
    session = Session(
        aws_access_key_id=os.environ.get("BEDROCK_IAM_ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("BEDROCK_IAM_SECRET_KEY"),
    )
    return session.client(
        service_name="bedrock-runtime",
        region_name="us-west-2",
        config=Config(retries={"max_attempts": 5, "mode": "standard"}),
    )


from langchain_aws.embeddings import BedrockEmbeddings
from pgvector_template.core.embedder import BaseEmbeddingProvider


class BedrockEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider for Amazon Bedrock"""

    def __init__(self, model_id: str = "amazon.titan-embed-text-v2:0", verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.verbose = verbose
        self.bedrock_client = _get_bedrock_client()
        self.bedrock_embeddings = BedrockEmbeddings(
            model_id=model_id,
            client=self.bedrock_client,
        )

    def embed_text(self, text: str) -> list[float]:
        """Get embedding for text"""
        vector = self.bedrock_embeddings.embed_documents([text])[0]
        if self.verbose:
            print(f"Embedding vector for '{text}': {vector}")
        return vector

    def embed_batch(self, texts) -> list[list[float]]:
        """Generate embeddings for multiple texts"""
        vectors = self.bedrock_embeddings.embed_documents(texts)
        if self.verbose:
            for i, text in enumerate(texts):
                print(f"Embedding vector for '{text}': {vectors[i]}")
        return vectors

    def get_dimensions(self):
        return 1024
