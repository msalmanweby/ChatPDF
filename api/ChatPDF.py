import os
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader 
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import CompositeElement, Table
from langchain.schema import Document 
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.retrievers import MultiVectorRetriever
from langchain.prompts import ChatPromptTemplate
import uuid
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from langchain.vectorstores import Chroma
from pathlib import Path

path = "test2.pdf"
persist_directory = "/home/codainer-b1/Workspace/CPPY/VectorStore"


def Pre_Processing():
    loader = UnstructuredFileLoader(path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter (
        chunk_size=2000, 
        chunk_overlap=200
    )

    document = text_splitter.split_documents(pages)

    return document


def Save_into_DB():
    os.environ["OPENAI_API_KEY"]= api_key
    embedding = OpenAIEmbeddings()
    document = Pre_Processing()

    vectorstore = Chroma.from_documents(document=document, embedding=embedding, persist_directory=persist_directory)
    retriever = vectorstore.as_retriever()

    return retriever


def Load_from_DB():
    os.environ["OPENAI_API_KEY"]= api_key
    embedding = OpenAIEmbeddings()

    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    retriever = vectorstore.as_retriever()

    return retriever


def SimpleMethod(self):

    # loader = UnstructuredFileLoader(path)
    # pages = loader.load()

    # text_splitter = RecursiveCharacterTextSplitter (
    #     chunk_size=2000, 
    #     chunk_overlap=200
    # )

    # document = text_splitter.split_documents(pages)

    # os.environ["OPENAI_API_KEY"]= api_key
    # embedding = OpenAIEmbeddings()

    # vectorstore = Chroma.from_documents(document=document, embedding_function=embedding, persist_directory=persist_directory)
    # retriever = vectorstore.as_retriever()

    retriever = Load_from_DB()

    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    def format_docs(pages):
        return "\n\n".join(page.page_content for page in pages)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    query = chain.invoke(self)

    return query


def ComplexMethod(self):
    raw_pdf_elements = partition_pdf(
    filename=path,
    extract_images_in_pdf=False,
    infer_table_structure=True,
    strategy="hi_res",
    #strategy="fast",#switch to this option, for simple pdf
    chunking_strategy="by_title",
    max_characters=1500,
    new_after_n_chars= 500,
    combine_text_under_n_chars=2000,
    image_output_dir_path=path
)
    # Create a dictionary to store counts of each type
    category_counts = {}

    for element in raw_pdf_elements:
        category = str(type(element))
        if category in category_counts:
            category_counts[category] += 1
        else:
            category_counts[category] = 1

    # Unique_categories will have unique elements
    unique_categories = set(category_counts.keys())
    print(category_counts)

    
    class Element(Document):
        type: str

    # Categorize by type
    categorized_elements = []
    for element in raw_pdf_elements:
        if isinstance(element, Table):
            categorized_elements.append(Element(type="table", page_content=str(element)))
        elif isinstance(element, CompositeElement):
            categorized_elements.append(Element(type="text", page_content=str(element)))

    # Tables
    table_elements = [e for e in categorized_elements if e.type == "table"]

    # Text
    text_elements = [e for e in categorized_elements if e.type == "text"]

    os.environ["OPENAI_API_KEY"]= api_key
    

    prompt_text = """You are an assistant tasked with summarizing tables and text. \
    Give a concise summary of the table or text. Table or text chunk: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    model = ChatOpenAI()
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    tables = [i.page_content for i in table_elements]
    table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

    texts = [i.page_content for i in text_elements]
    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
    

    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(collection_name="collection",
                        embedding_function=OpenAIEmbeddings())

    # The storage layer for the parent documents
    store = InMemoryStore()
    id_key = "id"

    # The retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # Add texts
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_texts = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(text_summaries)
    ]
    retriever.vectorstore.add_documents(summary_texts)
    retriever.docstore.mset(list(zip(doc_ids, texts)))

    # Add tables
    table_ids = [str(uuid.uuid4()) for _ in tables]
    summary_tables = [
        Document(page_content=s, metadata={id_key: table_ids[i]})
        for i, s in enumerate(table_summaries)
    ]
    if summary_tables:
        retriever.vectorstore.add_documents(summary_tables)
        retriever.docstore.mset(list(zip(table_ids, tables)))
    

    # Prompt template
    template = """Answer the question based only on the following context,
    which can include text and tables::
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # LLM
    model = ChatOpenAI(temperature=0.0)

    # RAG pipeline
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    query = chain.invoke(self)

    return query