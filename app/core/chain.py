"""
# TODO: Implement logic to return a configured LCEL chain
"""
from functools import lru_cache

from langchain_core.runnables import RunnableSerializable

from app.core.llm import *


from dotenv import load_dotenv
load_dotenv()


@lru_cache
def build_chain() -> RunnableSerializable:
    raise NotImplementedError("This function is not implemented yet.")
