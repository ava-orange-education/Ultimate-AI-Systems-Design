import numpy as np
from sklearn.metrics import mutual_info_score
from typing import List, Tuple

def calculate_mutual_information(x: List[int], y: List[int]) -> float:
    """
    Calculate mutual information between two variables.
    
    MI(X,Y) measures how much information X and Y share:
    - MI = 0: Variables are independent
    - Higher MI: Stronger relationship between variables
    
    Args:
        x: First variable's data
        y: Second variable's data
    
    Returns:
        float: Mutual information score
    """
    return mutual_info_score(x, y)

def interpret_mi_score(mi_score: float) -> str:
    """
    Interpret the mutual information score.
    
    Args:
        mi_score: Calculated mutual information score
    
    Returns:
        str: Interpretation of the score
    """
    if mi_score < 0.1:
        return "Very weak or no relationship"
    elif mi_score < 0.3:
        return "Weak relationship"
    elif mi_score < 0.5:
        return "Moderate relationship"
    else:
        return "Strong relationship"

def generate_example_scenarios() -> List[Tuple[str, List[int], List[int]]]:
    """
    Generate different scenarios to demonstrate mutual information.
    
    Returns:
        List of tuples containing (scenario name, x data, y data)
    """
    np.random.seed(42)
    
    scenarios = [
        # Perfect correlation (maximum MI)
        ("Perfect Correlation", 
         [0, 1, 0, 1, 0, 1, 0, 1],
         [0, 1, 0, 1, 0, 1, 0, 1]),
        
        # Perfect negative correlation (high MI)
        ("Perfect Negative Correlation",
         [0, 1, 0, 1, 0, 1, 0, 1],
         [1, 0, 1, 0, 1, 0, 1, 0]),
        
        # Independent variables (minimum MI)
        ("Independent Variables",
         [0, 0, 0, 0, 1, 1, 1, 1],
         [0, 1, 0, 1, 0, 1, 0, 1]),
        
        # Partial correlation (moderate MI)
        ("Partial Correlation",
         [0, 0, 1, 1, 0, 0, 1, 1],
         [0, 1, 1, 1, 0, 0, 1, 0]),
        
        # Noisy correlation (lower MI)
        ("Noisy Correlation",
         [0, 1, 0, 1, 0, 1, 0, 1],
         [0, 1, 1, 1, 0, 0, 0, 1]),
        
        # Random noise (very low MI)
        ("Random Noise",
         list(np.random.randint(0, 2, 8)),
         list(np.random.randint(0, 2, 8))),
    ]
    
    return scenarios

def analyze_relationship(x: List[int], y: List[int]) -> None:
    """
    Analyze the relationship between two variables.
    
    Args:
        x: First variable's data
        y: Second variable's data
    """
    # Calculate joint probability distribution
    unique_pairs = set(zip(x, y))
    joint_dist = {}
    n = len(x)
    
    for pair in unique_pairs:
        count = sum(1 for i in range(n) if x[i] == pair[0] and y[i] == pair[1])
        joint_dist[pair] = count / n
    
    print("\nJoint Distribution:")
    for pair, prob in joint_dist.items():
        print(f"P(X={pair[0]}, Y={pair[1]}) = {prob:.2f}")

def main():
    """
    Main function to demonstrate mutual information with various scenarios.
    """
    scenarios = generate_example_scenarios()
    
    print("Mutual Information Analysis")
    print("==========================")
    
    for scenario_name, x, y in scenarios:
        mi_score = calculate_mutual_information(x, y)
        interpretation = interpret_mi_score(mi_score)
        
        print(f"\nScenario: {scenario_name}")
        print("-" * (len(scenario_name) + 10))
        print(f"X: {x}")
        print(f"Y: {y}")
        print(f"Mutual Information Score: {mi_score:.3f}")
        print(f"Interpretation: {interpretation}")
        
        # Detailed analysis for each scenario
        analyze_relationship(x, y)

if __name__ == "__main__":
    main()
