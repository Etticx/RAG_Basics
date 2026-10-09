"""Basic thesis RAG, with retrieval evidence exposed for evaluation."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "gemini-3.6-flash"


def build_rag_chain(model=None):
    # Lazy imports let evaluation reports be plotted without starting the RAG.
    from dotenv import load_dotenv
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import Chroma
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    load_dotenv(BASE_DIR / ".env")
    print("Reading Documents...")
    docs = PyPDFLoader(str(BASE_DIR / "data" / "Thesis.pdf")).load()
    if not docs:
        raise ValueError("No document pages found in data/Thesis.pdf")
    print(f"Loaded {len(docs)} pages")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    ).split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    local_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=local_embeddings,
        persist_directory=str(BASE_DIR / "chroma_db"),
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatGoogleGenerativeAI(
        model=model or os.getenv("RAG_MODEL", DEFAULT_MODEL), temperature=0
    )
    # Preserve the original prompt so evaluation measures the existing behavior.
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that analyses documents and only
answer question based on the follow context:
{context}
Question: {question}
""")
    # Step 7: retrieve ONCE, and retain those same documents beside the answer.
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | RunnablePassthrough.assign(answer=prompt | llm | StrOutputParser())
    )


def answer_question(rag_chain, question):
    result = rag_chain.invoke(question)
    return {
        "question": question,
        "answer": result["answer"],
        "contexts": [doc.page_content for doc in result["context"]],
        "sources": [doc.metadata for doc in result["context"]],
    }


def main():
    rag_chain = build_rag_chain()
    print("\n--- System Ready ---\n")
    question = "what classifies an SME as a high risk?"
    result = answer_question(rag_chain, question)
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")


if __name__ == "__main__":
    main()
