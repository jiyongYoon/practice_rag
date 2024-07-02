from typing import List

from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
import os

load_dotenv()
__db_path = os.environ['DB_PATH']
__root_db_path = os.environ['PROJECT_ROOT_DB_PATH']


def save_text(embedding_model, text: List[str], collection_name: str = None):
    return Chroma.from_texts(
        text,
        embedding_model,
        persist_directory=__db_path,
        collection_name=collection_name,
    )


def save_document(embedding_model, document: List[Document], collection_name: str = None):
    return Chroma.from_documents(
        document,
        embedding_model,
        persist_directory=__db_path,
        collection_name=collection_name,
    )


def save_document_by_batch_size(embedding_model, document: List[Document], max_batch_size: int = 166, collection_name: str = None):
    def chunk_list(lst, chunk_size):
        """리스트를 일정 크기의 청크로 분할하는 함수."""
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]

    document_chunks = list(chunk_list(document, max_batch_size))

    for i, document_chunk in enumerate(document_chunks):
        print(f'Storing batch {i + 1}/{len(document_chunks)}')
        save_db = save_document(
            embedding_model=embedding_model,
            document=document_chunk,
            collection_name=collection_name,
        )


def find_db(embedding_model, collection_name: str = None):
    return Chroma(
        embedding_function=embedding_model,
        persist_directory=__db_path,
        collection_name=collection_name,
    )

def find_db_from_root(embedding_model, collection_name: str = None):
    return Chroma(
        embedding_function=embedding_model,
        persist_directory=__root_db_path,
        collection_name=collection_name,
    )


def get_all_documents(chroma_instance, collection_name: str):
    collection = chroma_instance._client.get_collection(collection_name)
    return collection.get()


def delete_from_ids(chroma_instance, ids: List[str], collection_name: str):
    chroma_instance.delete(ids=ids, collection_name=collection_name)


def delete_all_collection_element(chroma_instance, collection_name: str):
    documents = get_all_documents(chroma_instance, collection_name)
    all_ids = documents['ids']
    chroma_instance.delete(ids=all_ids, collection_name=collection_name)
    chroma_instance.delete_collection()
