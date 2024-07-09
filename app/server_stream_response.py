import asyncio
import os
from typing import AsyncIterable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.schema import HumanMessage
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles

from app.chat import chain as chat_chain
from dotenv import load_dotenv

load_dotenv()


app = FastAPI()

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

class Message(BaseModel):
    content: str


from langchain_community.chat_models import ChatOllama, ChatOpenAI


async def send_message(content: str) -> AsyncIterable[str]:
    callback = AsyncIteratorCallbackHandler()
    # model = ChatOllama(
    #     model="EEVE-Korean-10.8B:latest",
    #     streaming=True,
    #     verbose=True,
    #     callback=[callback],
    # )

    model = ChatOpenAI(
        streaming=True,
        verbose=True,
        callbacks=[callback],
        temperature=0.1,  # 창의성
        max_tokens=2048,  # 최대 토큰 수
        model="gpt-3.5-turbo",  # 모델명
        api_key=os.environ['OPEN_API_KEY'],
    )

    task = asyncio.create_task(
        model.agenerate(messages=[[HumanMessage(content=content)]])
    )

    try:
        async for token in callback.aiter():
            yield token
    except Exception as e:
        print(f"Caught exception: {e}")
    finally:
        callback.done.set()

    await task


@app.post("/stream_chat")
async def stream_chat(message: Message):
    print(message.content)
    generator = send_message(message.content)
    return StreamingResponse(generator, media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8484)