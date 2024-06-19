from langserve import RemoteRunnable
from dotenv import load_dotenv
import os


def generate_remote_llm():
    load_dotenv()

    llm_url = os.environ['NGROK_URL'] + '/llm'

    return RemoteRunnable(llm_url)
