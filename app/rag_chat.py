from vector_store import chroma_db
from model import openai_embeddings_model, ollama_llm_model
from langchain.chains import RetrievalQA
from langserve.pydantic_v1 import BaseModel, Field
from typing import List, Union
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


embedding_model = openai_embeddings_model.generate_embedding_model()
collection_name = 'safety_docs'
llm = ollama_llm_model.eeve_llm

find_db = chroma_db.find_db_from_root(
    embedding_model=embedding_model,
    collection_name=collection_name,
)

retriever = find_db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        'score_threshold': 0.75,
        'k': 5,
    }
)