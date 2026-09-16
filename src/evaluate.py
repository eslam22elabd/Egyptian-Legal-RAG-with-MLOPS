"""
USAGE:
    python -m src.evaluate --run-name "baseline-top3" --top-k 3
"""

import argparse
import json
import os
import time
from pathlib import Path

import mlflow

from src.retrieve import Retriever
from src.config import settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DATASET_PATH = PROJECT_ROOT / "eval" / "golden_dataset.json"


def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


def evaluate_retriever(retriever: Retriever, dataset: list[dict], top_k: int) -> dict:
    """Computes Recall@K, Hit Rate (= any expected article in top_k),
    and MRR (Mean Reciprocal Rank) across the golden dataset."""
    total = len(dataset)
    hits = 0
    reciprocal_ranks = []
    recall_sum = 0.0
    latencies_ms = []
    per_question_results = []

    for item in dataset:
        question = item["question"]
        expected = set(item["expected_article_numbers"])

        start = time.time()
        retrieved = retriever.retrieve(question, top_k=top_k)
        latencies_ms.append((time.time() - start) * 1000)

        retrieved_numbers = [r["article_number"] for r in retrieved]

        found_expected = [n for n in retrieved_numbers if n in expected]
        is_hit = len(found_expected) > 0
        hits += int(is_hit)

        recall = len(set(found_expected)) / len(expected) if expected else 0.0
        recall_sum += recall

        rank = None
        for idx, n in enumerate(retrieved_numbers, start=1):
            if n in expected:
                rank = idx
                break
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        per_question_results.append({
            "question": question,
            "expected": list(expected),
            "retrieved": retrieved_numbers,
            "hit": is_hit,
        })

    return {
        "hit_rate": hits / total,
        "recall_at_k": recall_sum / total,
        "mrr": sum(reciprocal_ranks) / total,
        "avg_latency_ms": sum(latencies_ms) / total,
        "total_questions": total,
        "per_question_results": per_question_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="eval-run")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("egyptian-legal-rag-retrieval")

    dataset = load_golden_dataset()
    retriever = Retriever()

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("embedding_model", settings.embedding_model)
        mlflow.log_param("top_k", args.top_k)
        mlflow.log_param("collection_name", settings.collection_name)
        mlflow.log_param("golden_dataset_size", len(dataset))

        results = evaluate_retriever(retriever, dataset, top_k=args.top_k)

        mlflow.log_metric("hit_rate", results["hit_rate"])
        mlflow.log_metric("recall_at_k", results["recall_at_k"])
        mlflow.log_metric("mrr", results["mrr"])
        mlflow.log_metric("avg_latency_ms", results["avg_latency_ms"])
        # Locally for you 
        artifact_path = PROJECT_ROOT / "mlruns_artifacts" / "last_run_details.json"
        artifact_path.parent.mkdir(exist_ok=True)
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(results["per_question_results"], f, ensure_ascii=False, indent=2)
        mlflow.log_artifact(str(artifact_path))

        print(f"hit_rate={results['hit_rate']:.2f} "
              f"recall_at_k={results['recall_at_k']:.2f} "
              f"mrr={results['mrr']:.2f} "
              f"avg_latency_ms={results['avg_latency_ms']:.1f}")


if __name__ == "__main__":
    main()