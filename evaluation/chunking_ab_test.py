"""Illustrative A/B comparison of chunking strategies using synthetic documents and
keyword-based recall estimates. This is not a measurement of production retrieval
quality. Run: python -m evaluation.chunking_ab_test"""

import asyncio
import logging
import random
import string
from dataclasses import dataclass

from ingestion.chunker import AdaptiveChunker

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class ChunkingExperimentResult:
    strategy: str
    n_chunks: float           # Average chunks per document.
    avg_chunk_len: float      # Average chunk length.
    simulated_recall: float   # Simulated Recall@5.
    coverage: float           # Content coverage estimate.


def simulate_recall(chunks, query_keywords: list[str], k: int = 5) -> float:
    """Approximate Recall@K using keyword matches in the highest-ranked chunks. A real
    retrieval evaluation requires embeddings and a vector index."""
    relevant_chunks = [
        c for c in chunks
        if any(kw.lower() in c.page_content.lower() for kw in query_keywords)
    ]
    if not relevant_chunks:
        return 1.0

    # Simulate retrieval of relevant chunks.
    # Keyword density approximates ranking.
    def keyword_density(chunk) -> float:
        text = chunk.page_content.lower()
        return sum(text.count(kw.lower()) for kw in query_keywords) / max(len(text), 1)

    sorted_chunks = sorted(chunks, key=keyword_density, reverse=True)
    top_k_chunks = sorted_chunks[:k]

    hit = sum(
        1 for rc in relevant_chunks
        if any(rc.page_content[:100] in tc.page_content or
               tc.page_content[:100] in rc.page_content
               for tc in top_k_chunks)
    )
    return min(hit / len(relevant_chunks), 1.0)


def run_chunking_experiment(
    sample_texts: list[str],
    strategies: list[str],
    query_keywords: list[str],
) -> list[ChunkingExperimentResult]:
    """Compare multiple chunking strategies."""
    chunker = AdaptiveChunker(chunk_size=512, chunk_overlap=64)
    results = []

    for strategy in strategies:
        all_chunks = []
        for text in sample_texts:
            chunks = chunker.chunk(text, metadata={"source": "test"}, strategy=strategy)
            all_chunks.extend(chunks)

        avg_chunk_len = (
            sum(len(c.page_content) for c in all_chunks) / len(all_chunks)
            if all_chunks else 0
        )
        avg_n_chunks = len(all_chunks) / len(sample_texts)

        # Average simulated recall across documents.
        recalls = []
        for text in sample_texts:
            chunks = chunker.chunk(text, metadata={"source": "test"}, strategy=strategy)
            recall = simulate_recall(chunks, query_keywords, k=5)
            recalls.append(recall)
        avg_recall = sum(recalls) / len(recalls) if recalls else 0

        # Estimate coverage after splitting.
        original_total = sum(len(t) for t in sample_texts)
        chunked_total = sum(
            len(chunker.chunk(t, strategy=strategy, metadata={}))
            for t in sample_texts
        )
        # Approximate coverage using chunk count.
        coverage = min(1.0, avg_n_chunks / 10)

        results.append(ChunkingExperimentResult(
            strategy=strategy,
            n_chunks=avg_n_chunks,
            avg_chunk_len=avg_chunk_len,
            simulated_recall=avg_recall,
            coverage=coverage,
        ))

    return results


def generate_sample_docs(n: int = 20, length: int = 3000) -> list[str]:
    """Generate synthetic enterprise documents with structured paragraphs."""
    topics = [
        "Employee handbook, Chapter 3: Leave policy.\n\nAnnual leave policy: employees receive 5 days of paid annual leave after one year of service, "
        "increasing to 10 days after three years and 15 days after ten years. Employees must apply two weeks in advance, "
        "and obtain approval from their direct supervisor. Annual leave cannot be carried over; unused leave expires at year-end.\n\n"
        "Sick leave policy: employees taking sick leave must provide a medical certificate. "
        "Sick leave exceeding three consecutive days requires a certificate from a tertiary Grade A hospital. Sick leave pay is 80% of base salary.\n\n"
        "Personal leave policy: up to three days requires department manager approval; longer leave requires HR approval. "
        "Personal leave is unpaid.",

        "IT support handbook: common troubleshooting guide.\n\n"
        "1. Cannot log into enterprise systems: first check the network connection and confirm the VPN is connected. "
        "If the issue persists, contact the IT helpdesk to reset the password.\n\n"
        "2. VPN connection failure: check the VPN client version and ensure it is up to date. "
        "Try another server. If the connection still fails, submit a support ticket.\n\n"
        "3. Email delivery failure: check the recipient address format and ensure attachments do not exceed 25MB. "
        "Clear the email client cache and retry.",

        "Fictional contract review sample (not legal guidance): confidentiality clauses.\n\n"
        "This sample requires confidentiality agreements to specify a confidentiality period. "
        "The sample recommends a period of no more than three years and treats longer periods as potentially unfair.\n\n"
        "Sample non-compete clause: the restriction generally lasts no more than two years, "
        "with compensation of at least 30% of average salary over the preceding twelve months.\n\n"
        "Sample penalty clause: the penalty should correspond to actual losses; "
        "excessive penalties may be adjusted by a court.",
    ]

    docs = []
    for i in range(n):
        base = topics[i % len(topics)]
        # Randomize synthetic document length.
        padding = " ".join(
            f"Additional clause {j}: refer to section {random.randint(1,50)} of the internal company policies."
            for j in range(length // 100)
        )
        docs.append(base + "\n\n" + padding)
    return docs


def print_experiment_results(results: list[ChunkingExperimentResult]):
    """Print the comparison table."""
    print("\n" + "="*65)
    print("  Chunking Strategy A/B Test Results")
    print("="*65)
    print(f"  {'Strategy':<20} {'Recall@5':>10} {'Avg Chunks':>12} {'Avg Len':>10}")
    print("-"*65)

    baseline_recall = None
    for r in results:
        if baseline_recall is None:
            baseline_recall = r.simulated_recall
            improvement = "baseline"
        else:
            pct = (r.simulated_recall - baseline_recall) / baseline_recall * 100
            improvement = f"+{pct:.0f}%" if pct > 0 else f"{pct:.0f}%"

        marker = " <- hierarchical example" if r.strategy == "hierarchical" else ""
        print(f"  {r.strategy:<20} {r.simulated_recall:>10.3f} "
              f"{r.avg_n_chunks:>12.1f} {r.avg_chunk_len:>10.0f}  "
              f"({improvement}){marker}")
    print("="*65 + "\n")


if __name__ == "__main__":
    print("Generating sample enterprise documents...")
    sample_docs = generate_sample_docs(n=20)
    query_keywords = ["annual leave", "policy", "employee", "apply", "approval"]

    print("Running A/B test across chunking strategies...")
    results = run_chunking_experiment(
        sample_texts=sample_docs,
        strategies=["fixed", "sentence", "hierarchical"],
        query_keywords=query_keywords,
    )

    print_experiment_results(results)
    print("Compare the simulated Recall@5 scores above.")
    print("      Hierarchical chunking combines child retrieval with parent context; validate on real data before production use.\n")
