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
test_questions = [
    {"q": "What is RAG?", "expected": "RAG is Retrieval-Augmented Generation"},
    {"q": "What is chunking?", "expected": "Chunking divides documents into smaller chunks"},
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