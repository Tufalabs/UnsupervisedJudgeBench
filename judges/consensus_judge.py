from typing import Tuple, List
from utils.inference import generate_text
import re
from judges.orm_judge import evaluate_solution as orm_evaluate

async def evaluate_solution_unanimous(
    question: str,
    solution: str,
    model_name: str = "gpt-4o-mini",
    num_votes: int = 3
) -> bool:
    """
    Evaluates the solution using multiple ORM judges that must unanimously agree.
    
    Args:
        question: The question to evaluate
        solution: The solution to evaluate
        model_name: The model to use for evaluation
        num_votes: Number of judges to use (default: 3)
        
    Returns:
        bool: True if all judges agree it's correct, False otherwise
    """
    evaluations = []
    
    # Get multiple evaluations
    for i in range(num_votes):
        result = await orm_evaluate(question, solution, model_name)
        evaluations.append(result)
    
    # All judges must agree it's correct
    return all(evaluations)

async def evaluate_solution_majority(
    question: str,
    solution: str,
    model_name: str = "gpt-4o-mini",
    num_votes: int = 3
) -> bool:
    """
    Evaluates the solution using multiple ORM judges with majority voting.
    
    Args:
        question: The question to evaluate
        solution: The solution to evaluate
        model_name: The model to use for evaluation
        num_votes: Number of judges to use (default: 3)
        
    Returns:
        bool: True if more than half of the judges agree it's correct, False otherwise
    """
    evaluations = []
    
    # Get multiple evaluations
    for i in range(num_votes):
        result = await orm_evaluate(question, solution, model_name)
        evaluations.append(result)
    
    # Count positive votes
    positive_votes = sum(1 for eval in evaluations if eval)
    
    # Majority means more than half
    return positive_votes > (num_votes / 2)

async def evaluate_solution(
    question: str,
    solution: str,
    model_name: str = "gpt-4o-mini",
    num_votes: int = 3,
    require_unanimous: bool = False
) -> bool:
    """
    Main evaluation function that uses either unanimous or majority voting.
    
    Args:
        question: The question to evaluate
        solution: The solution to evaluate
        model_name: The model to use for evaluation
        num_votes: Number of judges to use (default: 3)
        require_unanimous: Whether to require unanimous agreement (default: False)
        
    Returns:
        bool: True if the solution is correct according to the majority rule, False otherwise
    """
    if require_unanimous:
        return await evaluate_solution_unanimous(question, solution, model_name, num_votes)
    else:
        return await evaluate_solution_majority(question, solution, model_name, num_votes) 