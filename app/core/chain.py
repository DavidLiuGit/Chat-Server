"""
# TODO: Implement logic to return a configured LCEL chain
"""

from functools import lru_cache

from chat_chain.chain import ChatChain, ChatChainProps

from app.core.llm import *


@lru_cache
def build_chain() -> ChatChain:
    raise NotImplementedError("This function is not implemented yet.")
