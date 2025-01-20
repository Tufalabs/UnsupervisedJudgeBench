import json
import asyncio
from typing import Dict, List, Any
from pathlib import Path
from judges import orm_judge, prm_judge, orm_then_prm_judge, consensus_judge
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm

# ============ CONFIGURATION ============
NUM_SAMPLES = 100  # Number of samples to evaluate
JUDGE_TYPE = "orm"  # Options: "orm", "prm", "consensus"
MODEL_NAME = "gpt-4o"
# MODEL_NAME = "Qwen/QVQ-72B-Preview"
# To evaluate all judges
# JUDGE_TYPES = ["orm", "prm", "consensus"]

# Batch settings
MAX_BATCH_SIZE = 100  # Maximum concurrent evaluations

# Consensus judge settings (only used if JUDGE_TYPE = "consensus")
NUM_VOTES = 3
REQUIRE_UNANIMOUS = False
# =====================================

@dataclass
class EvalMetrics:
    total_samples: int = 0
    correct_predictions: int = 0
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    @property
    def accuracy(self) -> float:
        return self.correct_predictions / self.total_samples if self.total_samples > 0 else 0
    
    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator > 0 else 0
    
    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator > 0 else 0
    
    @property
    def f1_score(self) -> float:
        denominator = self.precision + self.recall
        return 2 * (self.precision * self.recall) / denominator if denominator > 0 else 0
    
    @property
    def false_positive_rate(self) -> float:
        denominator = self.false_positives + self.true_negatives
        return self.false_positives / denominator if denominator > 0 else 0
    
    def __str__(self) -> str:
        return f"""
Evaluation Metrics:
------------------
Total Samples: {self.total_samples}
Accuracy: {self.accuracy:.3f}
Precision: {self.precision:.3f}
Recall: {self.recall:.3f}
F1 Score: {self.f1_score:.3f}
False Positive Rate: {self.false_positive_rate:.3f}

Confusion Matrix:
---------------
True Positives: {self.true_positives}
True Negatives: {self.true_negatives}
False Positives: {self.false_positives}
False Negatives: {self.false_negatives}
"""

async def evaluate_batch(
    batch: List[Dict[str, Any]],
    judge,
    judge_type: str,
    model_name: str,
    metrics: EvalMetrics,
    source_metrics: Dict[str, EvalMetrics],
    pbar: tqdm
) -> None:
    """Evaluates a batch of samples concurrently"""
    tasks = []
    samples = []
    for sample in batch:
        if "is_correct" not in sample:
            continue
            
        task = asyncio.create_task(
            asyncio.wait_for(
                judge(sample["question"], sample["model_output"], MODEL_NAME),
                timeout=600
            )
        )
        tasks.append(task)
        samples.append(sample)
    
    # Run all tasks concurrently
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for sample, result in zip(samples, results):
            try:
                if isinstance(result, Exception):
                    print(f"\nError for {sample['source']}: {str(result)}")
                    continue
                    
                # All judges now return bool directly
                is_correct = result
                
                # Update metrics
                metrics.total_samples += 1
                source_metrics[sample["source"]].total_samples += 1
                
                if is_correct == sample["is_correct"]:
                    metrics.correct_predictions += 1
                    source_metrics[sample["source"]].correct_predictions += 1
                    
                if sample["is_correct"] and is_correct:
                    metrics.true_positives += 1
                    source_metrics[sample["source"]].true_positives += 1
                elif not sample["is_correct"] and not is_correct:
                    metrics.true_negatives += 1
                    source_metrics[sample["source"]].true_negatives += 1
                elif not sample["is_correct"] and is_correct:
                    metrics.false_positives += 1
                    source_metrics[sample["source"]].false_positives += 1
                else:  # sample["is_correct"] and not is_correct
                    metrics.false_negatives += 1
                    source_metrics[sample["source"]].false_negatives += 1
                
                pbar.update(1)
                
            except Exception as e:
                print(f"\nError processing result: {str(e)}")
                print(f"Result type: {type(result)}")
                print(f"Result value: {result}")
                continue
                
    except Exception as e:
        print(f"\nBatch processing error: {str(e)}")

async def evaluate_judge(
    samples: List[Dict[str, Any]], 
    judge_type: str = "orm",
    model_name: str = "gpt-4o-mini",
    num_votes: int = 3,
    require_unanimous: bool = False
) -> EvalMetrics:
    """
    Evaluates a judge's performance on the sample outputs.
    """
    metrics = EvalMetrics()
    source_metrics = defaultdict(EvalMetrics)
    
    # Select judge based on type
    if judge_type == "orm":
        judge = orm_judge.evaluate_solution
    elif judge_type == "prm":
        judge = prm_judge.evaluate_solution
    elif judge_type == "orm_then_prm":
        judge = orm_then_prm_judge.evaluate_solution
    elif judge_type == "consensus":
        judge = lambda q, s, m: consensus_judge.evaluate_solution(
            q, s, m, num_votes, require_unanimous
        )
    else:
        raise ValueError(f"Unknown judge type: {judge_type}")
    
    # Process samples in batches with progress bar
    batches = [samples[i:i + MAX_BATCH_SIZE] 
              for i in range(0, len(samples), MAX_BATCH_SIZE)]
    
    print(f"\nProcessing {len(samples)} samples in {len(batches)} batches")
    print(f"Batch size: {MAX_BATCH_SIZE}")
    
    with tqdm(total=len(samples), desc=f"Evaluating {judge_type.upper()} judge") as pbar:
        for i, batch in enumerate(batches):
            print(f"\nProcessing batch {i+1}/{len(batches)}")
            await evaluate_batch(
                batch, judge, judge_type, model_name, metrics, source_metrics, pbar
            )
    
    # Print per-source metrics
    print(f"\nMetrics by Source for {judge_type.upper()} Judge:")
    print("=" * 50)
    for source, source_metric in source_metrics.items():
        print(f"\nSource: {source}")
        print(source_metric)
    
    return metrics

async def evaluate_all_judges(samples: List[Dict[str, Any]]) -> Dict[str, Dict]:
    """Evaluates all judge types and returns combined results"""
    all_results = {}
    
    for judge_type in JUDGE_TYPES:
        print(f"\nEvaluating {judge_type.upper()} Judge")
        print("=" * 50)
        
        metrics = await evaluate_judge(
            samples,
            judge_type=judge_type,
            model_name=MODEL_NAME,
            num_votes=NUM_VOTES,
            require_unanimous=REQUIRE_UNANIMOUS
        )
        
        print(metrics)
        
        # Store results
        results = {
            "judge_type": judge_type,
            "model_name": MODEL_NAME,
            "num_votes": NUM_VOTES if judge_type == "consensus" else None,
            "require_unanimous": REQUIRE_UNANIMOUS if judge_type == "consensus" else None,
            "metrics": {
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "false_positive_rate": metrics.false_positive_rate,
                "confusion_matrix": {
                    "true_positives": metrics.true_positives,
                    "true_negatives": metrics.true_negatives,
                    "false_positives": metrics.false_positives,
                    "false_negatives": metrics.false_negatives
                }
            }
        }
        
        all_results[judge_type] = results
        
        # Save individual results
        output_file = f"results_{judge_type}_judge.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    # Save combined results
    with open("results_all_judges.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nCombined results saved to results_all_judges.json")
    
    return all_results

async def main():
    # Load sample outputs
    with open("sample_outputs.json", "r") as f:
        samples = json.load(f)
    
    if JUDGE_TYPE == "all":
        await evaluate_all_judges(samples)
    else:
        print(f"\nEvaluating {JUDGE_TYPE.upper()} Judge")
        print("=" * 50)
        
        metrics = await evaluate_judge(
            samples,
            judge_type=JUDGE_TYPE,
            model_name=MODEL_NAME,
            num_votes=NUM_VOTES,
            require_unanimous=REQUIRE_UNANIMOUS
        )
        
        print(metrics)
        
        # Save results to file
        results = {
            "judge_type": JUDGE_TYPE,
            "model_name": MODEL_NAME,
            "num_votes": NUM_VOTES if JUDGE_TYPE == "consensus" else None,
            "require_unanimous": REQUIRE_UNANIMOUS if JUDGE_TYPE == "consensus" else None,
            "metrics": {
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "false_positive_rate": metrics.false_positive_rate,
                "confusion_matrix": {
                    "true_positives": metrics.true_positives,
                    "true_negatives": metrics.true_negatives,
                    "false_positives": metrics.false_positives,
                    "false_negatives": metrics.false_negatives
                }
            }
        }
        
        output_file = f"results_{JUDGE_TYPE}_judge.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
