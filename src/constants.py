import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

# CONSTANT VARIABLES
_OPENAI_EMBEDDING_MODEL = OpenAIEmbeddings(
    openai_api_key=os.environ.get("<OPENAI_API_KEY>"),
)

GPT_4o = ChatOpenAI(
    openai_api_key=os.environ.get("<OPENAI_API_KEY>"),
    temperature=1e-10,
    max_tokens=500,
    model_name="gpt-4o"
)

rag_model = ChatOpenAI(
    openai_api_key=os.environ.get("<OPENAI_API_KEY>"),
    temperature=1e-10,
    max_tokens=500,
    model_name="gpt-3.5-turbo"
)