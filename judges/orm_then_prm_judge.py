from typing import Dict, Any
from utils.inference import generate_text
import re

async def evaluate_step(model_name, question, previous_steps, current_step):
    """Helper function to evaluate individual solution steps"""
    eval_prompt = f"""You are an expert Marker, evaluating at the level of a university professor. Focus on scoring reasoning, not phrasing. Your job is to mark the following step based on the question and previous steps provided.

Question:
{question}

Previous steps:
{previous_steps}

Current step:
{current_step}

Evaluate this step:
1. Is this step logically correct and relevant to solving the problem?
2. Does it follow from the previous steps?
3. Are there any errors in reasoning or calculation?

If a step is irrelevant, doesn't provide new information, or is just a restatement of the previous steps, give it a score of 0.

Only score it a -1 if it is mathmatically incorrect.

Finally, give a score for this step:
STEP SCORE: [1 if correct, 0 if neutral/irrelevant, -1 if incorrect]

Provide a brief explanation for your score."""

    evaluation = await generate_text(model=model_name, prompt=eval_prompt)
    score_match = re.search(r'STEP SCORE:\s*(-1|0|1)', evaluation)
    score = int(score_match.group(1)) if score_match else None
    
    return score, evaluation

async def evaluate_overall(model_name, question, solution):
    """Evaluates the entire solution in one go"""
    eval_prompt = f"""You are an expert Marker, evaluating at the level of a university professor. Your job is to determine if this solution correctly answers the given question. Follow these steps carefully:

Question:
{question}

Solution:
{solution}

First, let's analyze this solution step by step:
1. Break down the solution's approach and methodology
2. Verify each step's logical progression
3. Check for any calculation or reasoning errors
4. Ensure all parts of the question are addressed

Now, consider these key points:
- Is the solution's approach valid for this type of problem?
- Are all calculations and logical steps correct?
- Does it fully address all aspects of the question?
- Is the final answer complete and properly justified?

After careful consideration, provide your final verdict enclosed in tags:
<MARK>1</MARK> for correct solutions
<MARK>0</MARK> for incorrect solutions

Follow your mark with a detailed explanation of your reasoning."""

    evaluation = await generate_text(
        model=model_name, 
        prompt=eval_prompt,
        max_tokens=8000
    )
    score_match = re.search(r'<MARK>(0|1)</MARK>', evaluation)
    score = int(score_match.group(1)) if score_match else 0
    
    return score, evaluation

async def is_solution_correct(
    question: str, 
    solution: str, 
    use_prm: bool = True, 
    orm_model_name: str = "gpt-4o-mini",
    prm_model_name: str = "gpt-4o-mini"
) -> bool:
    """
    Checks if a solution properly answers its corresponding question.
    First uses ORM, then optionally uses PRM for detailed analysis.
    
    Args:
        question: The question to evaluate
        solution: The solution to evaluate
        use_prm: Whether to use PRM for detailed analysis
        orm_model_name: The model to use for overall evaluation (default: "gpt-4o-mini")
        prm_model_name: The model to use for step-by-step evaluation (default: "gpt-4o-mini")
    """
    if not question or not solution:
        return False

    # First, do the overall evaluation with ORM model
    overall_score, _ = await evaluate_overall(orm_model_name, question, solution)
    
    # If the solution is incorrect or PRM is not requested, return early
    if not overall_score or not use_prm:
        return bool(overall_score)

    # If solution is correct and PRM is requested, proceed with step-by-step evaluation
    steps = [step.strip() for step in solution.split('\n') if step.strip()]
    previous_steps = ""
    step_scores = []
    
    for step in steps:
        score, _ = await evaluate_step(prm_model_name, question, previous_steps, step)
        if score is not None:
            step_scores.append(score)
        previous_steps += step + "\n"
    
    has_incorrect_steps = any(score == -1 for score in step_scores)
    return not has_incorrect_steps


async def evaluate_solution(
    question: str,
    solution: str,
    model_name: str = "gpt-4o-mini"
) -> bool:
    return await is_solution_correct(question, solution, use_prm=True, prm_model_name=model_name)
    