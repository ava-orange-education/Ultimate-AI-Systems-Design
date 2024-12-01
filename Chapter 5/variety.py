# Import pandas library for data manipulation
import pandas as pd
# Note: numpy (np) is used but not imported - need to add import
import numpy as np

# Create a DataFrame with different types of data sources
data = pd.DataFrame({"data_type": ["image", "text", "tabular", "audio", "video"]})

# Calculate Shannon entropy to measure data variety:
# 1. value_counts(normalize=True) gets the proportion of each data type
# 2. np.log calculates natural logarithm
# 3. Multiply proportions by their logs and sum
# 4. Negative sign converts the result to positive entropy value
Shannon_entropy = -sum(data["data_type"].value_counts(normalize=True) * 
                      np.log(data["data_type"].value_counts(normalize=True)))

# Interpret the entropy value
def interpret_entropy(entropy, n_categories):
    # Maximum possible entropy for given number of categories
    max_entropy = np.log(n_categories)
    # Calculate percentage of maximum possible entropy
    entropy_percentage = (entropy / max_entropy) * 100
    
    # Interpretation logic
    if entropy == max_entropy:
        interpretation = "Perfect variety: All data types are equally represented"
    elif entropy_percentage >= 80:
        interpretation = "High variety: Data types are well-distributed"
    elif entropy_percentage >= 50:
        interpretation = "Moderate variety: Some data types are more common than others"
    else:
        interpretation = "Low variety: Data is dominated by few types"
        
    return {
        "entropy": round(entropy, 2),
        "max_possible": round(max_entropy, 2),
        "percentage": round(entropy_percentage, 2),
        "interpretation": interpretation
    }

# Get interpretation
result = interpret_entropy(Shannon_entropy, len(data))

# Print detailed analysis
print("\nData Variety Analysis:")
print(f"Shannon entropy: {result['entropy']}")
print(f"Maximum possible entropy: {result['max_possible']}")
print(f"Percentage of maximum: {result['percentage']}%")
print(f"Interpretation: {result['interpretation']}")