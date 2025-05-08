from app.core.chain import build_chain

async def handle_chat(body: dict):
    chain = build_chain()
    return await chain.ainvoke(body)

async def stream_chat(body: dict):
    chain = build_chain()
    async for chunk in chain.astream(body):
        yield f"data: {chunk}\n\n"
    yield "data: [DONE]\n\n"
