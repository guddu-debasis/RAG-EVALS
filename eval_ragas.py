import os
from dotenv import load
load()

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    AnswerRelevancyMetric
)
from deepeval.models.base_model import DeepEvalBaseLLM
from groq import Groq

# Groq ke liye Custom DeepEval Model class
class GroqLlama(DeepEvalBaseLLM):
    def __init__(self, model_name="openai/gpt-oss-20b"):
        self.model_name = model_name
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name

# 1. Test data
test_cases = [
    LLMTestCase(
        input="What is RAG?",
        actual_output="RAG stands for Retrieval-Augmented Generation. It combines retrieval with LLMs.",
        expected_output="RAG is Retrieval-Augmented Generation",
        retrieval_context=[
            "RAG stands for Retrieval-Augmented Generation. It is an AI technique that combines information retrieval with large language models (LLMs)."
        ]
    ),
    LLMTestCase(
        input="What is chunking?",
        actual_output="Chunking is dividing a large document into smaller segments.",
        expected_output="Chunking divides documents into smaller chunks",
        retrieval_context=[
            "Chunking is the process of dividing a large document into smaller text segments."
        ]
    )
]

# 2. Model instance
model = GroqLlama(model_name="openai/gpt-oss-20b")

# 3. Metrics
faithfulness = FaithfulnessMetric(model=model)
precision = ContextualPrecisionMetric(model=model)
recall = ContextualRecallMetric(model=model)
relevancy = AnswerRelevancyMetric(model=model)

# 4. Evaluate
results = evaluate(
    test_cases=test_cases,
    metrics=[faithfulness, precision, recall, relevancy]
)

print(results)