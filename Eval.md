# RAG Evaluation Report - RAG EVALS

> **Project:** Document Q&A with Reranker (Chroma + BGE Reranker + Groq gpt-oss-20b)
> **Evaluator:** Debasis
> **Date:** Oct 2025

## 1. Setup

**Stack:**
- **Loader:** TextLoader (`notes.txt`)
- **Splitter:** RecursiveCharacterTextSplitter (chunk_size=500, overlap=100)
- **Embeddings:** all-MiniLM-L6-v2
- **Vector DB:** Chroma (`./chroma_db`)
- **Retriever:** k=10 candidates
- **Reranker:** BAAI/bge-reranker-base (Top 3)
- **LLM:** Groq `openai/gpt-oss-20b` (temp=0)
- **Prompt:** "Answer using only provided context..."

**Test Questions:**
1. What is RAG?
2. What is chunking?

---

## 2. DeepEval + TypeSafe AI (jev-latest) - REAL RAG Results

This is the real evaluation where `app.py` was used live.

**Command:** `python eval_deepeval.py`

**Pipeline Tested:** `Chroma (10) -> Reranker (Top 3) -> Groq -> Jev Judge`

```
Reranker results for "What is RAG?":
Rank 1 | Score: 0.9996 - Perfect chunk
Rank 2 | Score: 0.8923 - Only heading "## 1. What is RAG?" (junk)
Rank 3 | Score: 0.1407 - Irrelevant

ANSWER: RAG stands for Retrieval-Augmented Generation...
```

**Final Scores:**

| Metric | Average Score | Pass Rate | Status |
| :--- | :--- | :--- | :--- |
| **Faithfulness** | **0.95** | 100% (2/2) | ✅ |
| **Answer Relevancy** | **0.95** | 100% (2/2) | ✅ |
| **Contextual Precision** | **0.68** | 100% (2/2) | ✅ |
| **Contextual Recall** | **0.97** | 100% (2/2) | ✅ |

**Performance:**
- Time Taken: **1.07s** for 2 test cases x 4 metrics = 8 evaluations
- Token Cost: **$0.00009 USD** (9e-05)
- Model: `jev-latest (TypeSafe AI)` - System One, no LLM call
- `input_tokens=~212, output_tokens=0 (not charged)`

**Key Insight from Jev Logs:**
> `1. Every factual claim... -> P(yes)=0.95, confidence=0.90 -> clearly holds`
> `How much grounded? -> Fully grounded (0.97)`

Precision 0.68 is low because Rank 2 chunk is just a heading. Reranker needs score threshold >0.5.

---

## 3. Fake vs Real Evaluation

| Run Type | Faithfulness | Precision | Meaning |
| :--- | :--- | :--- | :--- |
| **Fake (hardcoded strings)** | 0.78 | 0.90 | I manually wrote the answers |
| **Real (app.py live)** | **0.95** | **0.68** | My actual RAG chain answered |

Real is better for Faithfulness because my prompt `Do not invent facts` works.

---

## 4. RAGAS vs DeepEval (My Experience)

I tried both. Here is honest comparison:

### RAGAS (Traditional LLM-as-a-Judge)

**How it works:** Calls Groq/GPT 3-4 times per question, writes chain-of-thought, then scores.
**My Experience:**
- **Speed:** Very Slow. 1 question ~15-20 sec. 10 questions = 4-5 min. Sometimes timeout.
- **Cost:** High. 200-300 output tokens per metric. Bill goes high.
- **Accuracy:** Gives only number (0.89), no reason why.
- **Setup:** Heavy, needs langchain wrappers.

### DeepEval + Jev (TypeSafe AI)

**How it works:** System One probabilistic model. No essay writing, direct P(yes) probability with confidence.
**My Experience:**
- **Speed:** Bhayankar fast hai bhai! 0.1 sec per eval. Full suite in 1.07s. RAGAS se 100x fast.
- **Cost:** Token bachata hai bcz output token 0. My total cost $0.00009 - almost free. Jev does not charge output tokens.
- **Accuracy + Reasoning:** Best part. It tells WHY:
  > `Every claim supported? -> P=0.37, confidence=0.26, unclear - because 'for better retrieval' not in context`
  This helped me catch my reranker bug.
- **Probabilistic Layer:** Feels like a classifier, not a chatty LLM. `P(yes)=0.95, P(no)=0.05` with weights.
- **Setup:** Just `TypeSafeModel(model="jev-latest")` and `eval_mode="system_one"`

**Winner for me:** DeepEval + Jev. RAGAS is good for papers, but for daily dev loop, Jev is unbeatable.

> "jev bhai helpful hai probabilistic layer ki tarah token bhi bachata hai bcz output token 0, iddhar bhi kaam agaya"

---

## 5. Final Verdict

- My RAG is **95% faithful** - not hallucinating.
- **Contextual Recall 97%** - retrieving correct info.
- **Precision 68% needs fix** - need to filter reranker scores < 0.5 and clean `notes.txt` headings.
- For future, will use **DeepEval + Jev for CI** (fast + cheap) and **TruLens for dashboard** later.

**Next Steps:**
- [ ] Increase chunk_size from 500 to 800
- [ ] Add reranker threshold: `if score > 0.5`
- [ ] Add 20 more test questions
- [ ] Run `deepeval view` to save on Confident AI
- [ ] Push this report to GitHub

---
*Generated after live RAG evaluation with Jev-latest*
