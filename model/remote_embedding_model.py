from langserve import RemoteRunnable
from dotenv import load_dotenv
import os


### 아직 HuggingFaceEmbeddings 구현체에서 RemoteRunnable을 구현하여 지원하지 않음...
def generate_remote_embedding_model():
    load_dotenv()

    embedding_model_url = os.environ['NGROK_URL'] + '/embedding_model'

    return RemoteRunnable(embedding_model_url)