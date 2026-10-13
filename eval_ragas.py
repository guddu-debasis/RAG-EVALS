# eval_ragas.py
import time
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# tere app.py se import
from app import (
    load_documents,
    split_documents,
    create_vector_store,
    create_reranker,
    create_llm,
    create_prompt,
    answer_question
)

# 1. Judge LLM - Groq
groq_judge = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)
judge_llm = LangchainLLMWrapper(groq_judge)

# 2. Judge Embedding - local
hf_emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
judge_emb = LangchainEmbeddingsWrapper(hf_emb)

# 3. Test data - tere notes.txt se
questions = [
    "What is RAG?",
    "Why do we need RAG?",
    "What are main components of RAG?",
    "What is chunking?",
    "What is hybrid search?"
]

ground_truths = [
    "RAG combines retrieval with LLMs to answer using external docs",
    "LLM lacks private data, outdated knowledge, and hallucinates",
    "Document Loader, Text Splitter, Embedding Model, Vector DB, LLM",
    "Dividing large document into smaller chunks with overlap",
    "Combining keyword search with semantic vector search"
]

# 4. Init RAG
print("Init RAG...")
docs = load_documents()
chunks = split_documents(docs)
vs = create_vector_store(chunks)
retriever = vs.as_retriever(search_kwargs={"k": 10})
reranker = create_reranker()
llm = create_llm()
prompt = create_prompt()

answers = []
contexts = []

# 5. Run RAG for each question
for q in questions:
    print(f"\nAsking: {q}")
    result = answer_question(q, retriever, reranker, llm, prompt)
    
    # safety check
    if result is None:
        print("app.py return nahi kar raha, fix kar")
        ans = "No answer"
        ctx = ["No context"]
    else:
        ans, ctx = result
    
    answers.append(ans)
    contexts.append(ctx)
    time.sleep(2)

# 6. Build dataset
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}

dataset = Dataset.from_dict(data)

# 7. Evaluate
print("\nEvaluating with RAGAS...")
scores = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ],
    llm=judge_llm,
    embeddings=judge_emb
)

print(scores)