from openai.types.chat import ChatCompletion


def normalize_chat_completion(response: ChatCompletion) -> ChatCompletion:
    """
    Normalize ChatCompletion response to ensure OpenAI API compatibility.

    Converts content from list format to string format when needed.
    """
    for choice in response.choices:
        content = choice.message.content

        # If content is a list, extract text and concatenate
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            choice.message.content = "".join(text_parts)

    return response
