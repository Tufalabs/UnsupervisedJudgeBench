from typing import Tuple
from utils.inference import generate_text
import re
import asyncio
import json
from datetime import datetime
import os

async def evaluate_solution(
    question: str,
    solution: str,
    model_name: str = "gpt-4o-mini"
) -> Tuple[bool, str]:
    """
    Evaluates the entire solution in one go using overall reasoning.
    
    Args:
        question: The question to evaluate
        solution: The solution to evaluate
        model_name: The model to use for evaluation
        
    Returns:
        Tuple of (is_correct: bool, explanation: str)
    """
    eval_prompt = f"""You are an expert Marker, evaluating at the level of a university professor. Your job is to determine if this solution correctly answers the given question. Follow these steps carefully:

Question:
{question}

Solution:
{solution}

Where possible you don't need to resolve the problem. Just check if the solution is correct.
Do things like back substitution, or checking if the solution is correct. Running formal checks etc
Checking each step is correct etc

Don't just resolve the problem, just check if the solution is correct.

I want you to first reason through if the solution is correct. Then output your final verdict enclosed in tags:
<MARK>1</MARK> for correct solutions
<MARK>0</MARK> for incorrect solutions

Follow your mark with a detailed explanation of your reasoning."""


    evaluation = await generate_text(
        model=model_name, 
        prompt=eval_prompt,
        max_tokens=8000
    )
    score_match = re.search(r'<MARK>(0|1)</MARK>', evaluation)
    is_correct = bool(int(score_match.group(1))) if score_match else False
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Prepare log data
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'question': question,
        'solution': solution,
        'is_correct': is_correct,
        'explanation': evaluation,
        'model_name': model_name
    }
    
    # Generate unique filename with timestamp
    filename = f"logs/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Save to log file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    return is_correct
