import os
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

DOCUMENT_PATH = "notes.txt"
VECTOR_DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "BAAI/bge-reranker-base"
LLM_MODEL = "openai/gpt-oss-20b"
RETRIEVAL_K = 10
RERANK_TOP_K = 3

def load_documents():
    loader = TextLoader(DOCUMENT_PATH, encoding="utf-8")
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).")
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH,
    )
    print("Vector store is ready.")
    return vector_store

def create_reranker():
    print("Loading reranker model...")
    reranker = CrossEncoder(RERANKER_MODEL)
    print("Reranker is ready.")
    return reranker

def create_llm():
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is missing. Add it to your.env file.")
    return ChatGroq(model=LLM_MODEL, temperature=0)

def create_prompt():
    return ChatPromptTemplate.from_template("""
You are a helpful document question-answering assistant.

Answer the question using only the provided context.

If the context does not contain the answer, say:
"I don't know based on the provided document."

Do not invent facts that are not supported by the context.

Context:
{context}

Question:
{question}

Answer:
""")

def rerank_documents(question, candidates, reranker):
    if not candidates:
        return []
    pairs = [[question, doc.page_content] for doc in candidates]
    scores = reranker.predict(pairs)
    ranked_docs = sorted(
        zip(candidates, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    print("\nReranker results:")
    for rank, (doc, score) in enumerate(ranked_docs[:5], start=1):
        print(f"\nRank {rank} | Score: {float(score):.4f}")
        print(doc.page_content[:250])
    return [doc for doc, score in ranked_docs[:RERANK_TOP_K]]

def answer_question(question, retriever, reranker, llm, prompt):
    candidates = retriever.invoke(question)
    if not candidates:
        print("\nNo documents were retrieved.")
        return "No documents were retrieved.", []

    print(f"\nRetrieved {len(candidates)} candidate chunks.")
    selected_docs = rerank_documents(question, candidates, reranker)
    if not selected_docs:
        print("\nNo relevant chunks available.")
        return "No relevant chunks available.", []

    context = "\n\n".join(doc.page_content for doc in selected_docs)
    messages = prompt.invoke({"context": context, "question": question})
    response = llm.invoke(messages)

    print("\n" + "=" * 50)
    print("ANSWER")
    print("=" * 50)
    print(response.content)

    print("\n" + "=" * 50)
    print("FINAL CONTEXT SOURCES")
    print("=" * 50)
    for index, doc in enumerate(selected_docs, start=1):
        source = doc.metadata.get("source", "Unknown source")
        print(f"\nChunk {index} | Source: {source}")
        print(doc.page_content)

    # FIX: return for eval
    final_answer = response.content
    final_contexts = [doc.page_content for doc in selected_docs]
    return final_answer, final_contexts

def main():
    documents = load_documents()
    chunks = split_documents(documents)
    if not chunks:
        print("No text chunks found.")
        return
    vector_store = create_vector_store(chunks)
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    reranker = create_reranker()
    llm = create_llm()
    prompt = create_prompt()
    print("\nRAG application is ready!")
    while True:
        question = input("\nAsk a question (or type exit): ").strip()
        if question.lower() == "exit":
            print("Goodbye, bhai!")
            break
        if not question:
            continue
        try:
            answer_question(question, retriever, reranker, llm, prompt)
        except Exception as error:
            print(f"\nError processing question: {error}")

if __name__ == "__main__":
    main()