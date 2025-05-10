from langchain_core.messages import BaseMessage

from chat_chain.chain import ChatChain

from app.core.chain import build_chain
from app.models.v1.completions import Message, ChatCompletionsRequest


def handle_chat(request: ChatCompletionsRequest) -> Message:
    """
    Given a `ChatCompletionsRequest`, return a single `Message` from the `ChatChain`.
    """
    # fetch the existing chat
    chain = build_chain()

    user_input, chat_history = _structure_messages(request.messages)
    return Message(
        role="assistant",
        content=chain.chat(user_input, chat_history=chat_history),
    )


async def stream_chat(body: dict):
    chain = build_chain()
    async for chunk in chain.astream(body):
        yield f"data: {chunk}\n\n"
    yield "data: [DONE]\n\n"


def _structure_messages(messages: list[Message]) -> tuple[str, list[BaseMessage]]:
    """
    Given a list of `Message`s from the request, return:
    1. user's lastest input, as a string
    2. chat history , as `list[BaseMessage]`
    """
    return messages[-1].content, ChatChain.build_structured_chat_history(messages[:-1])
