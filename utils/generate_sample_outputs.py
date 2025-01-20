import asyncio
import json
import random
from typing import List, Dict, Tuple
from utils.inference import generate_text
from utils.get_dataset_question import (
    get_random_math_question,
    get_competition_math_problem,
    get_gpqa_question,
    get_numina_math_question,
    get_skunkworks_question,
    get_mathinstruct_question,
    get_numina_olympiad_question,
    get_aslawliet_olympiad_question,
    get_hard_competition_math_problem
)
from lib.supervised_eval import evaluate_text
from tqdm import tqdm

MODEL = "gpt-4o"

async def process_sample(
    source: str,
    get_question_func,
    prompt: str,
    model: str,
    semaphore: asyncio.Semaphore
) -> Tuple[Dict, str]:
    """Process a single sample with generation and evaluation"""
    problem, solution = get_question_func()
    full_prompt = f"{prompt}\nQuestion: {problem}\nPlease provide your final answer within <ANSWER></ANSWER> tags."
    
    question_data = {
        "source": source,
        "question": problem,
        "solution": solution
    }
    
    async with semaphore:
        try:
            model_output = await generate_text(
                prompt=full_prompt,
                model=model,
                max_tokens=4000
            )
            return question_data, model_output
        except Exception as e:
            print(f"Error generating output for {source}: {str(e)}")
            return question_data, ""

async def generate_samples(
    model: str,
    prompt: str,
    num_samples: int = 100,
    output_file: str = "sample_outputs.json",
    sources: List[str] = None,
    max_concurrent: int = 20
) -> List[Dict]:
    print(f"\nStarting generation of {num_samples} samples...")
    
    # Create semaphore for concurrent task limiting
    semaphore = asyncio.Semaphore(max_concurrent)

    # Define question sources with their corresponding functions
    question_sources = {
        "MATH": get_random_math_question,
        "Competition_Math": get_competition_math_problem,
        "GPQA": get_gpqa_question,
        "NuminaMath": get_numina_math_question,
        "Skunkworks": get_skunkworks_question,
        "MathInstruct": get_mathinstruct_question,
        "Numina_Olympiad": get_numina_olympiad_question,
        "Aslawliet_Olympiad": get_aslawliet_olympiad_question,
        "Competition_Math_Hard": get_hard_competition_math_problem
    }
    
    # If sources is provided, filter question_sources
    if sources:
        question_sources = {k: v for k, v in question_sources.items() if k in sources}
        print(f"Using sources: {', '.join(question_sources.keys())}")
        if not question_sources:
            raise ValueError("No valid sources provided. Available sources: " + 
                           ", ".join(question_sources.keys()))
    
    # Calculate samples per source
    samples_per_source = num_samples // len(question_sources)
    remaining_samples = num_samples % len(question_sources)
    
    # Create all tasks
    tasks = []
    for source, get_question_func in question_sources.items():
        num_source_samples = samples_per_source + (1 if remaining_samples > 0 else 0)
        remaining_samples -= 1 if remaining_samples > 0 else 0
        
        print(f"\nQueuing {num_source_samples} samples from {source}")
        source_tasks = [
            process_sample(source, get_question_func, prompt, model, semaphore)
            for _ in range(num_source_samples)
        ]
        tasks.extend(source_tasks)
    
    # Process all tasks concurrently with progress bar
    print("\nGenerating and evaluating outputs...")
    results = []
    with tqdm(total=len(tasks)) as pbar:
        for task in asyncio.as_completed(tasks):
            question_data, model_output = await task
            
            # Extract answer and evaluate
            extracted_answer = extract_answer(model_output)
            is_correct, _ = await evaluate_text(
                eval_model=MODEL,
                modelAnswer=extracted_answer,
                groundTruthAnswer=question_data["solution"]
            )
            
            # Create result entry
            result = {
                **question_data,
                "model_output": model_output,
                "extracted_answer": extracted_answer,
                "prompt": prompt,
                "model": model,
                "is_correct": bool(is_correct)
            }
            results.append(result)
            pbar.update(1)
            
            # Print progress for correct answers
            if result["is_correct"]:
                pbar.write(f"✓ Correct answer from {result['source']}")
    
    # Print results summary
    print(f"\nResults summary:")
    correct_count = sum(1 for result in results if result["is_correct"])
    print(f"Total correct: {correct_count}/{len(results)} ({correct_count/len(results)*100:.1f}%)")
    
    # Print per-source statistics
    for source in question_sources.keys():
        source_results = [r for r in results if r["source"] == source]
        source_correct = sum(1 for r in source_results if r["is_correct"])
        if source_results:
            print(f"{source}: {source_correct}/{len(source_results)} "
                  f"({source_correct/len(source_results)*100:.1f}% correct)")
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nGenerated {len(results)} samples and saved to {output_file}")
    return results

def extract_answer(text: str) -> str:
    """Extract answer from between tags"""
    import re
    match = re.search(r'<ANSWER>(.*?)</ANSWER>', text, re.DOTALL)
    return match.group(1).strip() if match else ""

# Example usage
async def main():
    prompt = "Solve the following math problem step by step:"
    samples = await generate_samples(
        model=MODEL,
        prompt=prompt,
        num_samples=100,
        output_file="sample_outputs.json",
        sources=["Competition_Math_Hard"],
        max_concurrent=100
    )

if __name__ == "__main__":
    asyncio.run(main())


