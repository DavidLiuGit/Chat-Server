"""
Here's the the bulk of the "magic" happens. Implement `build_chain()`.

`@lru_cache` is used to make the result effectively a singleton. Subsequent identical
calls to `build_chain()` will simply return the instance of the chain that already exists.
"""

from functools import lru_cache

from chat_chain.chain import ChatChain, ChatChainProps

from app.core.llm import get_language_model
from app.core.retriever import build_retriever


@lru_cache
def build_chain():
    """
    Build an instance of `ChatChainProps`
    """
    qa_llm = get_language_model()
    retriever = build_retriever()
    chat_chain_props = ChatChainProps(
        chat_llm=qa_llm,
        chat_prompt=(
            "You are a helpful assistant. Answer the following question based on the"
            " provided context. If you don't know the answer, just say that you don't know."
            " Don't make up an answer."
        ),
        retriever=retriever,
    )
    return ChatChain(chat_chain_props)
