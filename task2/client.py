import os

from openai import OpenAI
import chromadb


def get_openai_api_key():
    """Retrieve OPENAI API Key from environment variables."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
    return openai_api_key


openai_client = OpenAI(api_key=get_openai_api_key())
# check out
# https://realpython.com/chromadb-vector-database/
chroma_client = chromadb.PersistentClient(path='chroma_data/')
# https://docs.trychroma.com/reference/Client#httpclient
# https://docs.trychroma.com/usage-guide#running-chroma-in-clientserver-mode
# chroma_client = chromadb.HttpClient(host='localhost', port=8000)
