from dotenv import load_dotenv
load_dotenv()

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.models import TypeSafeModel

# Tera app.py import
from app import (
    load_documents, 
    split_documents, 
    create_vector_store, 
    create_reranker, 
    create_llm, 
    create_prompt, 
    answer_question
)

print("Setting up RAG...")
# Vector store load (agar chroma_db hai to wahi use hoga)
try:
    # Agar pehle se DB hai toh direct load
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    print("Loaded existing vector DB")
except:
    docs = load_documents()
    chunks = split_documents(docs)
    vector_store = create_vector_store(chunks)

retriever = vector_store.as_retriever(search_kwargs={"k": 10})
reranker = create_reranker()
llm = create_llm()
prompt = create_prompt()

# 2. Tere test questions
# test_questions = [
#     {"q": "What is RAG?", "expected": "RAG is Retrieval-Augmented Generation"},
#     {"q": "What is chunking?", "expected": "Chunking divides documents into smaller chunks"},
# ]

test_questions = [
    {"q": "What is RAG?", "expected": "RAG is Retrieval-Augmented Generation, combining retrieval with LLMs to answer using external documents like PDFs, text files, databases"},
    {"q": "What is chunking?", "expected": "Chunking divides documents into smaller chunks because full document is expensive and unnecessary for LLM"},
    {"q": "What are chunk size and chunk overlap?", "expected": "Chunk size is text length per chunk, chunk overlap is shared text between chunks to preserve context"},
    {"q": "What is an embedding model?", "expected": "Embedding model converts text chunks into vectors for similarity search, e.g., all-MiniLM-L6-v2"},
    {"q": "What is a vector database?", "expected": "Vector database stores embeddings and allows fast similarity search, e.g., Chroma DB in this app"},
    {"q": "What is a retriever?", "expected": "Retriever searches vector DB for top-k relevant chunks for a question, here k=10 candidates"},
    {"q": "What is reranking?", "expected": "Reranking reorders 10 candidate chunks by true relevance using BAAI/bge-reranker-base and selects top 3"},
    {"q": "What is the role of LLM in RAG?", "expected": "LLM generates natural language answer using question plus retrieved context, here Groq openai/gpt-oss-20b"},
    {"q": "What is a RAG prompt?", "expected": "RAG prompt tells LLM to answer using only provided context and say I don't know if answer not in context"},
    {"q": "What are 5 components of RAG?", "expected": "5 components are Document Loader, Text Splitter, Embedding Model, Vector Database and Retriever, Large Language Model"},
    {"q": "Why not pass whole document to LLM?", "expected": "Full document is expensive, exceeds context limit, and most parts are irrelevant for a specific question"},
    {"q": "What is RecursiveCharacterTextSplitter?", "expected": "RecursiveCharacterTextSplitter splits text using progressively smaller separators respecting chunk size"},
    {"q": "What is difference between embedding model and LLM?", "expected": "Embedding model finds relevant info via vectors, LLM generates answer using that info, different responsibilities"},
    {"q": "What are use cases of RAG?", "expected": "RAG useful for document assistants, customer support, university knowledge bases, legal search, internal assistants, tech docs chatbot"},
    {"q": "What is RAG evaluation?", "expected": "RAG evaluation checks Faithfulness, Answer Relevancy, Contextual Precision and Recall to measure hallucination and retrieval quality"},
    {"q": "What happens if context has no answer?", "expected": "If context does not contain answer, LLM should say I don't know based on provided document, not invent facts"},
    {"q": "What does temperature 0 mean?", "expected": "Temperature 0 means deterministic consistent answers with minimal randomness for factual Q&A"},
    {"q": "What is GROQ_API_KEY?", "expected": "GROQ_API_KEY is API key for Groq inference, needed in .env otherwise app raises ValueError"},
    {"q": "How does reranker score work?", "expected": "Reranker scores question-chunk pairs, high score like 0.9996 means highly relevant, low like 0.1 means irrelevant"},
    {"q": "What is final context source?", "expected": "Final context sources are top 3 reranked chunks from notes.txt joined and passed to LLM as context"},
]

test_cases = []
for item in test_questions:
    print(f"\n\n--- Testing: {item['q']} ---")
    # YEHI TERA APP.PY KA FUNCTION - LIVE CALL
    actual_answer, context_list = answer_question(item['q'], retriever, reranker, llm, prompt)
    
    # DeepEval test case banao
    test_cases.append(
        LLMTestCase(
            input=item['q'],
            actual_output=actual_answer,      # app.py se aaya
            expected_output=item['expected'],
            retrieval_context=context_list    # app.py ke reranked 3 chunks se aaya
        )
    )

# 3. Jev Judge
print("\n\n--- Evaluating with Jev ---")
jev = TypeSafeModel(model="jev-latest")
metrics = [
    FaithfulnessMetric(system_one_model=jev, eval_mode="system_one"),
    AnswerRelevancyMetric(system_one_model=jev, eval_mode="system_one"),
    ContextualPrecisionMetric(system_one_model=jev, eval_mode="system_one"),
    ContextualRecallMetric(system_one_model=jev, eval_mode="system_one"),
]

evaluate(test_cases=test_cases, metrics=metrics)