import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
groq_api_key = os.environ["GROQ_API_KEY"]

# Initialize documents, embeddings, and vector store
if "vector" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(
        st.session_state.docs[:20]
    )
    st.session_state.vectors = FAISS.from_documents(
        st.session_state.final_documents, st.session_state.embeddings
    )

# Title
st.title("ChatGroq Demo")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma-7b-it")

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

# Retrieval chain using Runnables
retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 5})

retrieval_chain = (
    {
        "context": RunnableLambda(lambda x: "\n\n".join(
            [doc.page_content for doc in retriever.invoke(x["input"])])),
        "input": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Streamlit UI
user_input = st.text_input("Input your prompt here")

if user_input:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_input})
    st.write("Response time:", time.process_time() - start)
    st.write(response)

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(retriever.invoke(user_input)):
            st.write(doc.page_content)
            st.write("--------------------------------")
