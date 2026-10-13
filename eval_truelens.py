import os
from dotenv import load_dotenv
load_dotenv()

from trulens.core import TruSession, Metric, Selector
from trulens.core.instruments import instrument
from trulens.providers.litellm import LiteLLM
from trulens.apps.app import TruApp
from trulens.dashboard import run_dashboard

from app import load_documents, split_documents, create_vector_store, create_reranker, create_llm, create_prompt

# 1. Groq provider via LiteLLM
groq_provider = LiteLLM("opensi/gpt-oss-20b")

# 2. Metrics setup using standard TruLens 2.x bindings
f_groundedness = Metric(implementation=groq_provider.groundedness_measure_with_cot_reasons)
f_relevance = Metric(implementation=groq_provider.relevance)

class RAGApp:
    def __init__(self):
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            self.vs = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        except:
            docs = load_documents()
            chunks = split_documents(docs)
            self.vs = create_vector_store(chunks)
        self.retriever = self.vs.as_retriever(search_kwargs={"k": 10})
        self.reranker = create_reranker()
        self.llm = create_llm()
        self.prompt = create_prompt()

    @instrument
    def retrieve(self, query: str) -> list:
        from app import rerank_documents
        cands = self.retriever.invoke(query)
        sel = rerank_documents(query, cands, self.reranker)
        return [d.page_content for d in sel]

    @instrument
    def generate_completion(self, query: str, context_str: list) -> str:
        context = "\n\n".join(context_str)
        msgs = self.prompt.invoke({"context": context, "question": query})
        return self.llm.invoke(msgs).content

    @instrument
    def query(self, query: str) -> str:
        context_str = self.retrieve(query=query)
        completion = self.generate_completion(query=query, context_str=context_str)
        return completion

session = TruSession()
rag = RAGApp()

tru_rag = TruApp(
    rag,
    app_name="RAG-Groq",
    app_version="v1",
    feedbacks=[f_groundedness, f_relevance]
)

with tru_rag as recording:
    print(rag.query("What is RAG?"))

run_dashboard(session)