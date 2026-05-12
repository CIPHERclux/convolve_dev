import asyncio
from app.services.llm_service import LLMService

async def test():
    llm = LLMService()
    try:
        res = await llm.generate_response(
            user_text="hello",
            conversation_history=[],
            biomarker_summary="",
            memory_context="",
            profile_summary="",
            safety=None
        )
        print("Success:", res)
    except Exception as e:
        print("Exception caught in script:", e)

asyncio.run(test())
