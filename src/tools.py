from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from src.constants import GPT_4o
from src.helper_functions import split_image_text_types, img_prompt_func


def multi_modal_rag_chain(retriever, model: ChatOpenAI = GPT_4o):
    """
    Multi-modal RAG chain
    """
    # RAG pipeline
    chain = (
            {
                "context": retriever | RunnableLambda(split_image_text_types),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(img_prompt_func)
            | model
            | StrOutputParser()
    )

    return chain
