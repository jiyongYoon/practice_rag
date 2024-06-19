from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base") # gpt 모델들에 대해서 토크나이징할 때 사용되는 임베딩 모델


def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)


def generate_text_splitter(chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=tiktoken_len,
    )
    return splitter


