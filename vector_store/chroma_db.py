from typing import List

from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
import os

load_dotenv()
__db_path = os.environ['DB_PATH']


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


def find_db(embedding_model, collection_name: str = None):
    return Chroma(
        embedding_function=embedding_model,
        persist_directory=__db_path,
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
