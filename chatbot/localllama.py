from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
import os  

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
#langsmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
    
##prompt template
prompt=ChatPromptTemplate.from_messages(
    [
    ("system","You are a helpful assistant. Please respond to the user query as best as you can."),
    ("user", "Question:{question}")

    ]
)

## streamlit framework
st.title("Chatbot with Langchain and Local Llama")
input_text=st.text_input("search the topic you want")

#Local Llama LLM
llm=Ollama(model="llama2")
output_parser=StrOutputParser()
chain=prompt | llm | output_parser  

if input_text:
    response=chain.invoke({"question":input_text})
    st.write(response)