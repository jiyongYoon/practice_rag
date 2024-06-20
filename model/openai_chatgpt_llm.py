from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


def generate_llm():
    load_dotenv()

    # 객체 생성
    llm = ChatOpenAI(
        temperature=0.1, # 창의성
        max_tokens=2048, # 최대 토큰 수
        model="gpt-3.5-turbo", # 모델명
        api_key=os.environ['OPEN_API_KEY'],
    )

    return llm


def generate_llm_streaming_ver():
    load_dotenv()

    llm = ChatOpenAI(
        temperature=0.1,  # 창의성
        max_tokens=2048,  # 최대 토큰 수
        model="gpt-3.5-turbo",  # 모델명
        callbacks=[StreamingStdOutCallbackHandler()], # 답변을 실시간으로
        api_key=os.environ['OPEN_API_KEY'],
    )

    return llm