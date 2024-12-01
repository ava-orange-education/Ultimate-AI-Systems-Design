import numpy as np
from scipy.stats import entropy
from typing import List, Union

def calculate_entropy(data: List[Union[int, float]]) -> float:
    """
    Calculate the Shannon entropy of a dataset.
    
    Shannon entropy measures the average information content or uncertainty in a dataset.
    Higher entropy indicates more uncertainty/randomness, lower entropy indicates more predictability.
    
    Args:
        data: List of numerical values (typically ratings or categorical data)
    
    Returns:
        float: Entropy value in bits (since base=2)
    
    Formula:
        H = -∑(p_i * log2(p_i)) where p_i is the probability of each unique value
    """
    # Count frequency of each unique value
    unique, counts = np.unique(data, return_counts=True)
    
    # Calculate probability distribution
    probabilities = counts / len(data)
    
    # Calculate entropy using scipy's entropy function with base 2 (bits)
    return entropy(probabilities, base=2)

def interpret_entropy(entropy_value: float) -> str:
    """
    Interpret the entropy value in terms of uncertainty and predictability.
    
    Interpretation scale:
    - < 1.0: Low entropy (highly predictable)
    - 1.0-2.0: Moderate entropy
    - > 2.0: High entropy (highly unpredictable)
    
    Args:
        entropy_value: Calculated entropy in bits
    
    Returns:
        str: Human-readable interpretation of the entropy value
    """
    if entropy_value < 1.0:
        return "Low Entropy (low uncertainty, high predictability)"
    elif entropy_value > 2.0:  
        return "High Entropy (high uncertainty, low predictability)"
    else:
        return "Moderate Entropy (moderate uncertainty and predictability)"

def analyze_distribution(data: List[Union[int, float]]) -> None:
    """
    Analyze the distribution of values and their entropy.
    
    Args:
        data: List of numerical values to analyze
    """
    unique, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)
    
    print("Distribution Analysis:")
    print("---------------------")
    for value, prob in zip(unique, probabilities):
        print(f"Value {value}: {prob:.2%}")
    print(f"Number of unique values: {len(unique)}")
    print(f"Most common value: {unique[np.argmax(counts)]}")
    print(f"Least common value: {unique[np.argmin(counts)]}")

def main():
    """
    Main function to demonstrate entropy calculation with various examples.
    """
    # Example 1: Perfect uniformity (maximum entropy)
    uniform_ratings = [1, 2, 3, 4, 5] * 2
    entropy_uniform = calculate_entropy(uniform_ratings)
    print("\nUniform Distribution:")
    print(f"Entropy: {entropy_uniform:.2f} bits")
    print(f"Interpretation: {interpret_entropy(entropy_uniform)}")
    analyze_distribution(uniform_ratings)

    print("\n" + "="*50 + "\n")

    # Example 2: Complete certainty (minimum entropy)
    certain_ratings = [5] * 10
    entropy_certain = calculate_entropy(certain_ratings)
    print("Complete Certainty:")
    print(f"Entropy: {entropy_certain:.2f} bits")
    print(f"Interpretation: {interpret_entropy(entropy_certain)}")
    analyze_distribution(certain_ratings)

    print("\n" + "="*50 + "\n")

    # Example 3: Realistic distribution
    realistic_ratings = [1, 2, 2, 3, 3, 3, 4, 5, 5, 5]
    entropy_realistic = calculate_entropy(realistic_ratings)
    print("Realistic Distribution:")
    print(f"Entropy: {entropy_realistic:.2f} bits")
    print(f"Interpretation: {interpret_entropy(entropy_realistic)}")
    analyze_distribution(realistic_ratings)

if __name__ == "__main__":
    main()