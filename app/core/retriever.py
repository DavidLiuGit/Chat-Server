import os

from langchain_core.retrievers import BaseRetriever

from langchain_logseq.loaders.journal_filesystem_loader import LogseqJournalFilesystemLoader
from langchain_logseq.loaders.journal_loader_input import LogseqJournalLoaderInput
from langchain_logseq.retrievers.contextualizer import (
    RetrieverContextualizer,
    RetrieverContextualizerProps,
)
from langchain_logseq.retrievers.journal_date_range_retriever import LogseqJournalDateRangeRetriever
from langchain_logseq.retrievers.pgvector_journal_retriever import PGVectorJournalRetriever
from langchain_logseq.models.journal_pgvector import JournalDocument
from langchain_logseq.uploaders.pgvector.journal_corpus_manager import JournalCorpusManager
from pgvector_template.db.document_db import DocumentDatabaseManager
from pgvector_template.service import DocumentService, DocumentServiceConfig
from pgvector_template.core import SearchQuery  # this should be customized in Langchain Logseq

from app.core.llm import get_fast_language_model, BedrockEmbeddingProvider


def build_retriever() -> BaseRetriever:
    """
    Build a `Retriever` to be used to contexualize the chat.
    """
    return _build_pgvector_journal_retriever()
    # llm = get_fast_language_model()
    # journal_path = os.environ.get("LOGSEQ_JOURNAL_PATH", "./journals")
    # loader = LogseqJournalFilesystemLoader(journal_path)
    # contextualizer = RetrieverContextualizer(
    #     RetrieverContextualizerProps(
    #         llm=llm,
    #         prompt=(
    #             "Given the user_input, and optional chat_history, create an query object based"
    #             "on the schema provided, if you believe it is relevant. Do not include anything"
    #             "except for the schema, serialized as JSON. Do not answer the question directly"
    #             " and do not include any preamble or additional context in your response."
    #         ),
    #         output_schema=LogseqJournalLoaderInput,
    #         enable_chat_history=True,
    #     )
    # )
    # return LogseqJournalDateRangeRetriever(
    #     contextualizer,
    #     loader,
    # )


def _build_pgvector_journal_retriever() -> BaseRetriever:
    """
    Build a `Retriever` backed by a `RetrieverContextualizer` operating a PGVector database
    containing Logseq journal entries.
    """

    # Database for PGVector retriever
    db_url = _get_pgvector_url()
    db_manager = DocumentDatabaseManager(
        database_url=db_url, schema_suffix="logseq", document_classes=[JournalDocument]
    )
    db_manager.setup()
    # should not need to call db_manager.setup(), unless it has not been set up yet

    # LLM & Embedder
    llm = get_fast_language_model()
    embedding_provider = BedrockEmbeddingProvider()

        
    # set up RetrieverContextualizer
    contextualizer = RetrieverContextualizer(
        RetrieverContextualizerProps(
            llm=llm,
            prompt=(
                "Given the user_input, and optional chat_history, create a search query object based "
                "on the schema provided, if you believe it is relevant. Do not include anything "
                "except for the schema, serialized as JSON. Do not answer the question directly"
            ),
            output_schema=SearchQuery,
            enable_chat_history=True,
        )
    )
    
    with db_manager.get_session() as session:
        # set up DocumentService, which provides the querying client
        doc_service_cfg = DocumentServiceConfig(
            document_cls=JournalDocument,
            embedding_provider=embedding_provider,
            corpus_manager_cls=JournalCorpusManager,
        )
        document_service = DocumentService(session, doc_service_cfg)
        
        return PGVectorJournalRetriever(
            contextualizer=contextualizer,
            document_service=document_service,
        )


def _get_pgvector_url():
    db_username = os.getenv("PGVECTOR_USERNAME")
    db_password = os.getenv("PGVECTOR_PASSWORD")
    db_host = os.getenv("PGVECTOR_HOST", "localhost")
    db_port = os.getenv("PGVECTOR_PORT", "5432")
    db = os.getenv("PGVECTOR_DB", "postgres")
    db_url = f"postgresql+psycopg://{db_username}:{db_password}@{db_host}:{db_port}/{db}"
    if not db_url:
        raise ValueError("PGVECTOR_DB_URL environment variable is not set")
    return db_url
