# Multi-Modal RAG

## Overview
Multi-Modal RAG (Retrieval-Augmented Generation) is an advanced retrieval system combining both textual and visual data for enhanced content understanding and question-answering. This project utilizes a multi-modal approach to retrieve relevant information from both text documents and images. The dataset used in this system is a Macroeconomics 101 course, which includes a textbook PDF and images embedded within the document.

## Dataset
The dataset is sourced from a Macroeconomics 101 course and consists of two primary components:

1. **[Images](https://github.com/dfavenfre/MultiModal-RAG/tree/main/src/dataset/images):** Visual elements embedded within the course material, including graphs, charts, and illustrations.
2. **[Documents](https://github.com/dfavenfre/MultiModal-RAG/blob/main/src/dataset/documents/macroeconomics_101.pdf):** A PDF file that contains the textual content for the course, covering fundamental macroeconomic concepts.

## Preprocessing

### Document Preprocessing
The text from the document is extracted using a PDF loader that parses each page of the macroeconomics course material. The following function extracts the text:

```python
def load_and_create_document(file_path: str) -> List[str]:
  """
  Loads a PDF document from a given file path and
  returns a list of strings containing the text content of each page.
  """
  loader = PyPDFLoader(file_path)
  docs = loader.load()
  texts = [d.page_content for d in docs]

  return texts
```

### Image-to-Text Preprocessing
Images and tables are summarized to be used in the retrieval process. The text summaries are designed to be concise and optimized for retrieval.

```Python
def generate_text_summaries(
    texts: List[str],
    tables: Optional[List] = [],
    summarize_texts: bool=True,
    model: ChatOpenAI = GPT_4o
    ):
  """
  Summarize text elements
  texts: List of str
  tables: List of str
  summarize_texts: Bool to summarize texts
  """

  # Prompt
  prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval. \
  These summaries will be embedded and used to retrieve the raw text or table elements. \
  Give a concise summary of the table or text that is well optimized for retrieval. Table or text: {element} """
  prompt = PromptTemplate.from_template(prompt_text)
  empty_response = RunnableLambda(
      lambda x: AIMessage(content="Error processing document")
  )

  # Text summary chain
  summarize_chain = (
      {
          "element": lambda x: x
      }
      | prompt
      | model
      | StrOutputParser()
  )

  # Initialize empty summaries
  text_summaries = []
  table_summaries = []

  # Apply to text if texts are provided and summarization is requested
  if texts and summarize_texts:
      text_summaries = summarize_chain.batch(texts, {"max_concurrency": 1})
  elif texts:
      text_summaries = texts

  # Apply to tables if tables are provided
  if tables:
    table_summaries = summarize_chain.batch(tables, {"max_concurrency": 1})

    return text_summaries, table_summaries

  else:
    return text_summaries
```

## MultiModal Vectorstore Creation
To store and retrieve both text and images, a ChromaDB vectorstore is created, which handles embedding and similarity search.
### Create ChromaDB Vectorstore with documents
```Python
def create_chroma_vectorstore(
    collection_name: Optional[str] = "mm_rag_for_econ_101",
    directory_name: Optional[str] = './chromadb'
    ) -> Chroma:
    """
    Description:
    -------------
    Creates a multi-vector store with the given retriever,
    collection name, and directory name.

    Args:
        retriever: The multi-vector retriever to use.
        collection_name: The name of the collection to create.
        directory_name: The directory to store the collection in.

    Returns:
        A Chroma object representing the multi-vector store.
    """
    vectorstore = Chroma(
    collection_name=collection_name,
    embedding_function=_OPENAI_EMBEDDING_MODEL,
    persist_directory=directory_name,
    collection_metadata={"hnsw:space":"cosine"}
    )
    vectorstore.persist()

    return vectorstore
```
## Update multi-index (text|image|table) vectorstore on existing ChromaDB 
New documents (text or images) can be added to the ChromaDB vectorstore for ongoing updates.

```Python
def update_documents(
    retriever: Chroma,
    doc_summaries: List[str],
    doc_contents: List[str]
    ) -> None:
    """Adds documents to a MultiVectorRetriever.

    Args:
        retriever: The chroma vectorstore instance.
        doc_summaries: A list of document summaries.
        doc_contents: A list of document contents.
    """

    doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
    summary_docs = [
        Document(page_content=s, metadata={"doc_id": doc_ids[i]})
        for i, s in enumerate(doc_summaries)
    ]
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, doc_contents)))
```
## Create Multi-modal Vectorstore
A multi-modal retriever is created that can handle text, tables, and images in the vectorstore. This allows retrieval across multiple modalities.
```Python
def create_multi_vector_retriever(
    vectorstore,
    text_summaries: List[str],
    texts: List[str],
    image_summaries,
    images,
    table_summaries: Optional[List[str]] = None,
    tables: Optional[List[str]] = None,
    ):
  """
  Create retriever that indexes summaries, but returns raw images or texts
  """

  # Initialize the storage layer
  store = InMemoryStore()
  id_key = "doc_id"

  # Create the multi-vector retriever
  retriever = MultiVectorRetriever(
      vectorstore=vectorstore,
      docstore=store,
      id_key=id_key,
  )

  # Check that text_summaries is not empty before adding
  if text_summaries:
      update_documents(retriever, text_summaries, texts)
  # Check that table_summaries is not empty before adding
  if table_summaries:
      update_documents(retriever, table_summaries, tables)
  # Check that image_summaries is not empty before adding
  if image_summaries:
      update_documents(retriever, image_summaries, images)

  return retriever 
```


## Evaluation
The system's performance is evaluated by issuing prompts and measuring the retrieval latency, the output image, and its description.
|Prompt| Output Image | Image Description | Retrieval Latency|
|-|-|-|-|
|Show the chart where the relationship between CPI and GDP deflator used for calculating inflation rate| ![image](https://github.com/user-attachments/assets/0b119956-ed06-466e-89fc-1aef8dd9a1e6) | The chart provided shows the relationship between the Consumer Price Index (CPI) and the GDP deflator in calculating the inflation rate from 1960 to 2010. The inflation rates computed using either the CPI or the GDP deflator are largely similar, as indicated by the close movement of the two lines over the years. The CPI is represented by the blue line, while the GDP deflator is represented by the pink line.|  0:00:05.817202|


