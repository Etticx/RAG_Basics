#0 import libraries
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import TextLoader #for loading text files
from langchain_community.document_loaders import PyPDFLoader #for loading pdf files
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI #embedding needs paid API key
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI #rate limiting
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

#1 Load data 
print("Reading Documents...")
loader = PyPDFLoader("data/Thesis.pdf")
docs = loader.load()
if (docs == None):
    print("No documents found")
    exit()
else:
    print(f"Loaded {len(docs)} Pages documents")


#2 Split data into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100
)
chunks = text_splitter.split_documents(docs)
if (chunks == None):
    print("No Chunks found")
else:
    print(f"Split into {len(chunks)} chunks")

#3,4  Create embeddings and store in ChromaDB
print("Creating embeddings and creating vector database...")
local_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorestore = Chroma.from_documents(
    documents = chunks,
    ##embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2"),
    embedding = local_embeddings,
    persist_directory = "./chroma_db"
)

#5 Retrieval: to set up retriever to find the top 3 most relevant chunks for a given query
retriever = vectorestore.as_retriever(search_kwargs={"k": 5})

#6 Setup chatbot and LLM
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
system_message = """
You are a helpful assistant that analyses documents and only 
answer question based on the follow context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(system_message)

#7 creates RAG chain via LangChain Expression Language (LCEL) where it chains all the steps altogether
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

#8 testing
print("\n--- System Ready ---\n")
user_query = "what classifies an SME as a high risk?"
print(f"Question: {user_query}")
response = rag_chain.invoke(user_query)
print(f"Answer: {response}")