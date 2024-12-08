from scipy.stats import entropy
import numpy as np
from typing import List, Tuple

def calculate_kl_divergence(P: List[float], Q: List[float]) -> float:
    """
    Calculate Kullback-Leibler divergence between two distributions P and Q.
    KL(P||Q) measures how much P differs from Q.
    
    Args:
        P: Observed distribution
        Q: Expected/reference distribution
    """
    return entropy(P, Q)

def interpret_kl_divergence(kl_value: float) -> str:
    """Interpret KL divergence value"""
    if kl_value < 0.05:
        return "Very similar distributions"
    elif kl_value < 0.2:
        return "Moderately different distributions"
    elif kl_value < 0.5:
        return "Substantially different distributions"
    else:
        return "Very different distributions"

def generate_example_scenarios() -> List[Tuple[str, List[float], List[float]]]:
    """Generate example scenarios to demonstrate KL divergence"""
    scenarios = [
        ("Identical Distributions",
         [0.25, 0.25, 0.25, 0.25],
         [0.25, 0.25, 0.25, 0.25]),
        
        ("Slightly Different",
         [0.3, 0.3, 0.2, 0.2],
         [0.25, 0.25, 0.25, 0.25]),
        
        ("Moderately Different",
         [0.4, 0.4, 0.1, 0.1],
         [0.25, 0.25, 0.25, 0.25]),
        
        ("Very Different",
         [0.7, 0.1, 0.1, 0.1],
         [0.25, 0.25, 0.25, 0.25]),
        
        ("Real-world Example (Product Ratings)",
         [0.5, 0.3, 0.1, 0.1],  # Observed ratings distribution
         [0.2, 0.3, 0.3, 0.2])  # Expected balanced ratings
    ]
    return scenarios

def main():
    print("KL Divergence Analysis Examples")
    print("==============================")
    
    scenarios = generate_example_scenarios()
    for name, P, Q in scenarios:
        kl_value = calculate_kl_divergence(P, Q)
        interpretation = interpret_kl_divergence(kl_value)
        
        print(f"\nScenario: {name}")
        print(f"Observed (P): {[f'{p:.2f}' for p in P]}")
        print(f"Expected (Q): {[f'{q:.2f}' for q in Q]}")
        print(f"KL(P||Q): {kl_value:.4f}")
        print(f"Interpretation: {interpretation}")
        
        # Show practical meaning
        if name == "Real-world Example (Product Ratings)":
            print("\nPractical Meaning:")
            print("- Observed distribution shows more 5-star ratings than expected")
            print("- Could indicate rating bias or genuine product quality")
            print("- Higher KL divergence suggests significant deviation from balanced ratings")

if __name__ == "__main__":
    main()
