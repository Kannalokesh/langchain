
#  Imports

import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings  
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


#  Environment setup

load_dotenv()
st.title("Hugging RAG")


#  Step 1 — Load PDFs

loader = PyPDFDirectoryLoader("./research_papers")  # folder path
documents = loader.load()


# Step 2 — Split into chunks

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(documents)



# Step 3 — Create embeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Build FAISS vectorstore
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# Step 4 — Load Phi-2 model (local)

import torch
device = 0 if torch.cuda.is_available() else -1
hf = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-base",
    task="text2text-generation",
    device=device,
    model_kwargs={"torch_dtype": "auto"},
    pipeline_kwargs={"temperature": 0.2, "max_new_tokens": 300}
)


# Step 5 — Prompt Template

prompt = ChatPromptTemplate.from_template("""
Answer the following question using the given context. 
If the answer cannot be found, reply with "Information not found in the context."

<context>
{context}
</context>

Question: {question}
""")


#  Step 6 — Build Retrieval Chain (Runnable version)

retrieval_chain = (
    {
        "context": RunnableLambda(
            lambda x: "\n\n".join(
                [doc.page_content for doc in retriever.invoke(x["question"])]
            )
        ),
        "question": RunnablePassthrough()
    }
    | prompt
    | hf
    | StrOutputParser()
)


#  Step 7 — Streamlit UI

user_query = st.text_input("## Hey Buddy, how can i help you?")

if user_query:
    with st.spinner("Generating answer..."):
        response = retrieval_chain.invoke({"question": user_query})
        st.markdown("### Response:")
        st.write(response)
