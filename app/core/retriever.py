import os

from langchain_core.retrievers import BaseRetriever

from langchain_logseq.loaders.journal_filesystem_loader import LogseqJournalFilesystemLoader
from langchain_logseq.loaders.journal_loader_input import LogseqJournalLoaderInput
from langchain_logseq.retrievers.contextualizer import RetrieverContextualizer, RetrieverContextualizerProps
from langchain_logseq.retrievers.journal_date_range_retriever import LogseqJournalDateRangeRetriever

from app.core.llm import get_fast_language_model


def build_retriever() -> BaseRetriever:
    """
    Build a `Retriever` to be used to contexualize the chat.
    """
    llm = get_fast_language_model()
    journal_path = os.environ.get("LOGSEQ_JOURNAL_PATH", "./journals")
    loader = LogseqJournalFilesystemLoader(journal_path)
    contextualizer = RetrieverContextualizer(
        RetrieverContextualizerProps(
            llm=llm,
            prompt=(
                "Given the user_input, and optional chat_history, create an query object based"
                "on the schema provided, if you believe it is relevant. Do not include anything"
                "except for the schema, serialized as JSON. Do not answer the question directly"
            ),
            output_schema=LogseqJournalLoaderInput,
            enable_chat_history=True,
        )
    )
    return LogseqJournalDateRangeRetriever(
        contextualizer,
        loader,
    )
