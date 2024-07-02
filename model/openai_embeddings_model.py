from dotenv import load_dotenv
import os
from langchain_openai.embeddings import OpenAIEmbeddings


def generate_embedding_model():
    load_dotenv()

    embeddings_model = OpenAIEmbeddings(
        api_key=os.environ['OPEN_API_KEY'],
        show_progress_bar=True,
    )

    return embeddings_model