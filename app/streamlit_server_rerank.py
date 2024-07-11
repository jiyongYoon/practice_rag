import os
import streamlit as st
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import ChatMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.vectorstores.faiss import FAISS
from langserve import RemoteRunnable
from langchain_openai import ChatOpenAI
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import List
from langchain_core.documents import Document

from model import ollama_llm_model, openai_embeddings_model
from splitter import text_splitter
from app.rag_chat import retriever
from retriever import rerank_retriever

# ⭐️ Embedding 설정
# USE_BGE_EMBEDDING = True 로 설정시 HuggingFace BAAI/bge-m3 임베딩 사용 (2.7GB 다운로드 시간 걸릴 수 있습니다)
# # USE_BGE_EMBEDDING = False 로 설정시 OpenAIEmbeddings 사용 (OPENAI_API_KEY 입력 필요. 과금)
# USE_BGE_EMBEDDING = False
#
# if not USE_BGE_EMBEDDING:
#     # OPENAI API KEY 입력
#     # Embedding 을 무료 한글 임베딩으로 대체하면 필요 없음!
#     os.environ["OPENAI_API_KEY"] = "OPENAI API KEY 입력"

# ⭐️ LangServe 모델 설정(EndPoint)
# 1) REMOTE 접속: 본인의 REMOTE LANGSERVE 주소 입력
# (예시)
# LANGSERVE_ENDPOINT = "https://poodle-deep-marmot.ngrok-free.app/llm/"
LANGSERVE_ENDPOINT = "https://cd4c-220-75-173-230.ngrok-free.app/llm/"

# 2) LocalHost 접속: 끝에 붙는 N4XyA 는 각자 다르니
# http://localhost:8000/llm/playground 에서 python SDK 에서 확인!
# LANGSERVE_ENDPOINT = "http://localhost:8000/llm/c/N4XyA"

# # 필수 디렉토리 생성 @Mineru
# if not os.path.exists(".cache"):
#     os.mkdir(".cache")
# if not os.path.exists(".cache/embeddings"):
#     os.mkdir(".cache/embeddings")
# if not os.path.exists(".cache/files"):
#     os.mkdir(".cache/files")

# 프롬프트를 자유롭게 수정해 보세요!
RAG_PROMPT_TEMPLATE = """
당신은 질문에 친절하게 답변하는 AI 입니다. 검색된 다음 문맥을 사용하여 질문에 답하세요. 답을 모른다면 모른다고 답변하세요.
답변을 할때는 근거의 출처를 답변 뒤에 붙여주세요. 모든 답변에는 반드시 출처를 함께 답변해야합니다.
근거의 출처는 context에 함께 전달되는 metadata 필드 안에있는 정보입니다.
파일이름은 'source'의 .pdf 확장자 앞에있는 이름이고, 페이지는 'page'의 숫자에 1을 더하세요.

출력 형태는 

[출처]
파일이름: 파일이름.pdf, 페이지: 15)

와 같이 해주면 됩니다.
여러 파일에서 근거를 찾았다면 위 형태를 활용해서 여러개를 모두 명시해주세요.

Question: {question} 
Context: {context} 
Answer:"""

st.set_page_config(page_title="safely 챗봇 테스트", page_icon="💬")
st.title("safely 챗봇 테스트")


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(role="assistant", content="무엇을 도와드릴까요?")
    ]


def print_history():
    for msg in st.session_state.messages:
        st.chat_message(msg.role).write(msg.content)


def add_history(role, content):
    st.session_state.messages.append(ChatMessage(role=role, content=content))


def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)


def plain_docs(docs):
    return docs


def format_docs_with_name_and_page(docs: List[Document]) -> str:
    formatted_texts = []

    for doc in docs:
        file_name = doc.metadata.get('source', 'unknown').split('/')[-1]
        page_number = doc.metadata.get('page', 'unknown') + 1  # 페이지 번호가 0부터 시작하는 경우 +1
        page_content = doc.page_content

        formatted_text = f"파일명: {file_name}, 페이지: {page_number}\n내용:\n{page_content}\n"
        formatted_texts.append(formatted_text)

    return "\n".join(formatted_texts)


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./safety_docs/pdf/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=500,
    #     chunk_overlap=50,
    #     separators=["\n\n", "\n", "(?<=\. )", " ", ""],
    #     length_function=len,
    # )
    splitter = text_splitter.generate_text_splitter(2000, 200)
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = openai_embeddings_model.generate_embedding_model()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, embedding=cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


with st.sidebar:
    file = st.file_uploader(
        "파일 업로드",
        type=["pdf", "txt", "docx"],
    )

# if file:
#     # retriever = embed_file(file)
#     retriever = retriever

print_history()


if user_input := st.chat_input():
    add_history("user", user_input)
    st.chat_message("user").write(user_input)
    with st.chat_message("assistant"):
        # ngrok remote 주소 설정
        # eeve_llm = RemoteRunnable(LANGSERVE_ENDPOINT)
        eeve_llm = ollama_llm_model.eeve_llm
        # LM Studio 모델 설정
        # ollama = ChatOpenAI(
        #     base_url="http://localhost:1234/v1",
        #     api_key="lm-studio",
        #     model="teddylee777/EEVE-Korean-Instruct-10.8B-v1.0-gguf",
        #     streaming=True,
        #     callbacks=[StreamingStdOutCallbackHandler()],  # 스트리밍 콜백 추가
        # )
        chat_container = st.empty()
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        rerank_retriever = rerank_retriever.load_rerank_retriever(retriever)


        # 체인을 생성합니다.
        rag_chain = (
                {
                    # "context": retriever | format_docs,
                    "context": rerank_retriever | format_docs_with_name_and_page,
                    "question": RunnablePassthrough(),
                }
                | prompt
                | eeve_llm
                | StrOutputParser()
        )
        # 문서에 대한 질의를 입력하고, 답변을 출력합니다.
        answer = rag_chain.stream(user_input)  # 문서에 대한 질의
        chunks = []
        for chunk in answer:
            chunks.append(chunk)
            chat_container.markdown("".join(chunks))
        add_history("ai", "".join(chunks))
