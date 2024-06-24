from vector_store import chroma_db
from model import openai_embeddings_model

collection_name = 'safely_4-1-bad'

find_db = chroma_db.find_db(
    openai_embeddings_model.generate_embedding_model(),
    collection_name=collection_name
)

print(len(find_db))

chroma_db.delete_all_collection_element(find_db, collection_name)

find_db = chroma_db.find_db(
    openai_embeddings_model.generate_embedding_model(),
    collection_name=collection_name
)

print(len(find_db))
