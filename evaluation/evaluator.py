"""RAG evaluation utilities for Recall@K, MRR@K, NDCG@K, judged hallucination rate, and
answer relevance. Usage: create RAGEvaluator(agent, llm_judge_client), await
run_eval(test_dataset), then call print_report(results)."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    """A ground-truth question-answer evaluation sample."""
    query: str
    ground_truth_answer: str
    relevant_doc_ids: list[str]       # Relevant document IDs.
    tenant_id: str = "eval_tenant"
    use_case: str = "kb_qa"


@dataclass
class EvalResult:
    """Evaluation results for one sample."""
    query: str
    predicted_answer: str
    retrieved_doc_ids: list[str]
    recall_at_5: float
    mrr_at_5: float
    is_hallucination: bool            # True means unsupported by context.
    answer_relevance_score: float     # Normalized judge score.
    latency_ms: int


@dataclass
class EvalReport:
    """Aggregate evaluation metrics."""
    n_samples: int
    recall_at_5: float
    mrr_at_5: float
    ndcg_at_5: float
    hallucination_rate: float         # Flagged samples divided by sample count.
    avg_answer_relevance: float
    avg_latency_ms: float
    p95_latency_ms: float


class RAGEvaluator:
    """End-to-end RAG evaluation. Run retrieval and generation, compare retrieved documents
    with ground truth, and use an LLM judge for groundedness and answer relevance."""

    def __init__(self, agent, openai_api_key: str):
        self.agent = agent
        self.judge = AsyncOpenAI(api_key=openai_api_key)

    # ─────────────────────────────────────────
    # Evaluation entry point.
    # ─────────────────────────────────────────
    async def run_eval(
        self,
        dataset: list[EvalSample],
        concurrency: int = 5,  # Limit concurrent API requests.
    ) -> EvalReport:
        """Evaluate a dataset. concurrency controls simultaneous sample evaluations."""
        semaphore = asyncio.Semaphore(concurrency)
        tasks = [
            self._eval_single(sample, semaphore)
            for sample in dataset
        ]
        results: list[EvalResult] = await asyncio.gather(*tasks)

        return self._aggregate(results)

    async def _eval_single(
        self,
        sample: EvalSample,
        semaphore: asyncio.Semaphore,
    ) -> EvalResult:
        """Evaluate one sample."""
        async with semaphore:
            import time
            t0 = time.perf_counter()

            # Run the agent.
            response = await self.agent.run(sample.query, sample.tenant_id)
            latency_ms = int((time.perf_counter() - t0) * 1000)

            retrieved_ids = [
                s.get("source", "") for s in response.get("sources", [])
            ]
            predicted_answer = response.get("answer", "")
            context_texts = [
                s.get("content", "") for s in response.get("sources", [])
            ]

            # Retrieval metrics.
            recall = self._recall_at_k(retrieved_ids, sample.relevant_doc_ids, k=5)
            mrr = self._mrr_at_k(retrieved_ids, sample.relevant_doc_ids, k=5)

            # Judge groundedness and relevance concurrently.
            hallucination_task = asyncio.create_task(
                self._judge_hallucination(
                    sample.query, predicted_answer, context_texts
                )
            )
            relevance_task = asyncio.create_task(
                self._judge_relevance(sample.query, predicted_answer)
            )
            is_hallucination, relevance_score = await asyncio.gather(
                hallucination_task, relevance_task
            )

            return EvalResult(
                query=sample.query,
                predicted_answer=predicted_answer,
                retrieved_doc_ids=retrieved_ids,
                recall_at_5=recall,
                mrr_at_5=mrr,
                is_hallucination=is_hallucination,
                answer_relevance_score=relevance_score,
                latency_ms=latency_ms,
            )

    # ─────────────────────────────────────────
    # Groundedness judge.
    # ─────────────────────────────────────────
    async def _judge_hallucination(
        self,
        query: str,
        answer: str,
        context_texts: list[str],
    ) -> bool:
        """Ask an LLM whether the answer is supported by the context. Return True for an
        answer judged unsupported."""
        context = "\n\n".join(f"[文档{i+1}]: {t}" for i, t in enumerate(context_texts))
        prompt = f"""你是一个严格的RAG质量评估员。

问题：{query}

检索到的文档：
{context}

模型回答：
{answer}

请判断：模型回答中的关键信息是否都有上述文档的支撑？
- 如果回答包含文档中没有的信息（即幻觉），回复：HALLUCINATION
- 如果回答完全基于文档内容，回复：GROUNDED
只回复一个词，不要其他内容。"""

        resp = await self.judge.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        verdict = resp.choices[0].message.content.strip().upper()
        return "HALLUCINATION" in verdict

    # ─────────────────────────────────────────
    # Relevance judge.
    # ─────────────────────────────────────────
    async def _judge_relevance(self, query: str, answer: str) -> float:
        """Score answer relevance from 0 to 10 using an LLM and normalize to 0 to 1."""
        prompt = f"""请评估以下回答对于给定问题的相关性和有用性。

问题：{query}
回答：{answer}

评分标准（0-10分）：
- 10分：完全回答问题，信息准确完整
- 7-9分：基本回答问题，有少量遗漏
- 4-6分：部分相关，但不够全面或有偏差
- 1-3分：回答与问题相关性低
- 0分：完全不相关或拒绝回答

只回复一个数字（0-10），不要其他内容。"""

        try:
            resp = await self.judge.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.0,
            )
            score_str = resp.choices[0].message.content.strip()
            score = float(score_str) / 10.0
            return max(0.0, min(1.0, score))
        except (ValueError, Exception):
            return 0.5  # Neutral score if parsing fails.

    # ─────────────────────────────────────────
    # Retrieval metric calculations.
    # ─────────────────────────────────────────
    @staticmethod
    def _recall_at_k(
        retrieved: list[str], relevant: list[str], k: int = 5
    ) -> float:
        """Compute recall among the first K retrieved documents."""
        if not relevant:
            return 1.0
        retrieved_k = set(retrieved[:k])
        relevant_set = set(relevant)
        return len(retrieved_k & relevant_set) / len(relevant_set)

    @staticmethod
    def _mrr_at_k(
        retrieved: list[str], relevant: list[str], k: int = 5
    ) -> float:
        """Compute the reciprocal rank of the first relevant document."""
        relevant_set = set(relevant)
        for rank, doc_id in enumerate(retrieved[:k], 1):
            if doc_id in relevant_set:
                return 1.0 / rank
        return 0.0

    # ─────────────────────────────────────────
    # Aggregate report.
    # ─────────────────────────────────────────
    @staticmethod
    def _aggregate(results: list[EvalResult]) -> EvalReport:
        if not results:
            raise ValueError("No evaluation results to aggregate.")

        latencies = np.array([r.latency_ms for r in results])
        return EvalReport(
            n_samples=len(results),
            recall_at_5=round(np.mean([r.recall_at_5 for r in results]), 4),
            mrr_at_5=round(np.mean([r.mrr_at_5 for r in results]), 4),
            ndcg_at_5=round(np.mean([r.recall_at_5 for r in results]), 4),  # Simplified approximation.
            hallucination_rate=round(
                sum(r.is_hallucination for r in results) / len(results), 4
            ),
            avg_answer_relevance=round(
                np.mean([r.answer_relevance_score for r in results]), 4
            ),
            avg_latency_ms=round(float(np.mean(latencies)), 1),
            p95_latency_ms=round(float(np.percentile(latencies, 95)), 1),
        )

    def print_report(self, report: EvalReport):
        """Print a formatted evaluation report."""
        print("\n" + "="*55)
        print("  RAG Evaluation Report")
        print("="*55)
        print(f"  Samples evaluated:     {report.n_samples}")
        print(f"  Recall@5:              {report.recall_at_5:.3f}")
        print(f"  MRR@5:                 {report.mrr_at_5:.3f}")
        print(f"  NDCG@5:                {report.ndcg_at_5:.3f}")
        print(f"  Hallucination rate:    {report.hallucination_rate:.1%}")
        print(f"  Avg answer relevance:  {report.avg_answer_relevance:.3f}")
        print(f"  Avg latency:           {report.avg_latency_ms:.0f}ms")
        print(f"  P95 latency:           {report.p95_latency_ms:.0f}ms")
        print("="*55 + "\n")
