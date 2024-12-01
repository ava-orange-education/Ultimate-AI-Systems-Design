"""
Data Quality Utility Calculator

This module provides functionality to calculate and analyze utility trade-offs
between data quality, volume, and cost. It implements a configurable framework
for decision-making in data collection and quality management scenarios.

Key Features:
- Configurable weights for quality, volume, and cost
- Component-wise utility analysis
- Decision support through interpretation
- Scenario comparison capabilities
"""

from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Tuple
from enum import Enum

class DecisionCategory(Enum):
    """Categories for decision interpretation"""
    OPTIMAL = "Optimal Balance"
    QUALITY_DRIVEN = "Quality Driven"
    VOLUME_DRIVEN = "Volume Driven"
    COST_SENSITIVE = "Cost Sensitive"
    SUBOPTIMAL = "Suboptimal"

@dataclass
class UtilityConfig:
    """Configuration parameters for utility calculation
    
    Attributes:
        alpha (float): Quality weight (0-1)
        beta (float): Volume weight (0-1)
        gamma (float): Cost weight (0-1)
        
    Note: Weights must sum to 1
    """
    alpha: float  # Quality weight
    beta: float   # Volume weight
    gamma: float  # Cost weight

class SimpleUtilityCalculator:
    """Calculator for data quality utility optimization"""
    
    def __init__(self, config: UtilityConfig):
        """Initialize calculator with weights
        
        Args:
            config: UtilityConfig with weights for components
            
        Raises:
            ValueError: If weights don't sum to 1
        """
        self.config = config
        self._validate_weights()
        
    def _validate_weights(self) -> None:
        """Validate that configuration weights sum to 1"""
        total = self.config.alpha + self.config.beta + self.config.gamma
        if not np.isclose(total, 1.0):
            raise ValueError(f"Weights must sum to 1, got {total}")

    def quality_benefit(self, q: float) -> float:
        """Calculate quality benefit using linear function
        
        Args:
            q: Quality score (0-1)
            
        Returns:
            Quality benefit score
            
        Raises:
            ValueError: If quality score is outside [0,1]
        """
        if not 0 <= q <= 1:
            raise ValueError(f"Quality must be between 0 and 1, got {q}")
        return q  # Linear relationship

    def volume_benefit(self, v: int) -> float:
        """Calculate volume benefit with diminishing returns
        
        Args:
            v: Number of data points
            
        Returns:
            Normalized volume benefit score
        """
        return np.log1p(v) / np.log1p(1e6)  # Normalized log scale

    def cost_function(self, q: float, v: int) -> float:
        """Calculate cost based on quality and volume
        
        Args:
            q: Quality score
            v: Volume of data
            
        Returns:
            Normalized cost score
        """
        return (q * v) / 1e6  # Simple normalized cost

    def calculate_utility(self, quality: float, volume: int) -> Dict[str, float]:
        """Calculate overall utility and component values
        
        Args:
            quality: Data quality score (0-1)
            volume: Number of data points
            
        Returns:
            Dictionary containing:
                - utility: Overall utility score
                - quality_contribution: Quality component
                - volume_contribution: Volume component
                - cost_impact: Cost component
        """
        q_benefit = self.config.alpha * self.quality_benefit(quality)
        v_benefit = self.config.beta * self.volume_benefit(volume)
        cost = self.config.gamma * self.cost_function(quality, volume)
        
        utility = q_benefit + v_benefit - cost
        
        return {
            'utility': utility,
            'quality_contribution': q_benefit,
            'volume_contribution': v_benefit,
            'cost_impact': cost
        }

    def interpret_result(self, result: Dict[str, float]) -> Dict[str, str]:
        """Interpret utility calculation results for decision making
        
        Args:
            result: Dictionary containing utility components
            
        Returns:
            Dictionary containing interpretations and recommendations
        """
        # Extract components
        utility = result['utility']
        q_contrib = result['quality_contribution']
        v_contrib = result['volume_contribution']
        cost = result['cost_impact']
        
        # Determine dominant factor
        components = {
            'quality': q_contrib,
            'volume': v_contrib,
            'cost': cost
        }
        dominant = max(components.items(), key=lambda x: abs(x[1]))
        
        # Categorize decision
        if utility > 0.7:
            category = DecisionCategory.OPTIMAL
        elif dominant[0] == 'quality' and q_contrib > 0.5:
            category = DecisionCategory.QUALITY_DRIVEN
        elif dominant[0] == 'volume' and v_contrib > 0.5:
            category = DecisionCategory.VOLUME_DRIVEN
        elif dominant[0] == 'cost' and cost > 0.3:
            category = DecisionCategory.COST_SENSITIVE
        else:
            category = DecisionCategory.SUBOPTIMAL
        
        # Generate recommendations
        recommendations = []
        if cost > 0.3:
            recommendations.append("Consider cost optimization")
        if q_contrib < 0.3:
            recommendations.append("Quality improvement recommended")
        if v_contrib < 0.2:
            recommendations.append("Consider increasing data volume")
            
        return {
            'category': category.value,
            'dominant_factor': f"{dominant[0].title()} driven",
            'utility_assessment': self._assess_utility(utility),
            'recommendations': recommendations if recommendations else ["Maintain current balance"],
            'efficiency_ratio': utility / cost if cost > 0 else float('inf')
        }

    def _assess_utility(self, utility: float) -> str:
        """Assess utility score and provide interpretation
        
        Args:
            utility: Calculated utility score
            
        Returns:
            String interpretation of utility score
        """
        if utility > 0.8:
            return "Excellent utility balance"
        elif utility > 0.6:
            return "Good utility balance"
        elif utility > 0.4:
            return "Moderate utility"
        else:
            return "Needs improvement"

def main():
    """Example usage of utility calculator with interpretation"""
    # Initialize with quality-focused configuration
    config = UtilityConfig(
        alpha=0.5,  # High weight on quality
        beta=0.3,   # Moderate weight on volume
        gamma=0.2   # Lower weight on cost
    )
    
    calculator = SimpleUtilityCalculator(config)
    
    # Test scenarios
    scenarios = [
        {'quality': 0.8, 'volume': 10000},  # Balanced scenario
        {'quality': 0.9, 'volume': 5000},   # Quality-focused scenario
        {'quality': 0.7, 'volume': 20000}   # Volume-focused scenario
    ]
    
    print("\nUtility Analysis Results:")
    print("=" * 50)
    
    for i, scenario in enumerate(scenarios, 1):
        result = calculator.calculate_utility(**scenario)
        interpretation = calculator.interpret_result(result)
        
        print(f"\nScenario {i}: Q={scenario['quality']}, V={scenario['volume']}")
        print("-" * 30)
        print(f"Total Utility: {result['utility']:.4f}")
        print("\nComponents:")
        print(f"  Quality: {result['quality_contribution']:.4f}")
        print(f"  Volume:  {result['volume_contribution']:.4f}")
        print(f"  Cost:    {result['cost_impact']:.4f}")
        
        print("\nInterpretation:")
        print(f"  Category: {interpretation['category']}")
        print(f"  Assessment: {interpretation['utility_assessment']}")
        print(f"  Dominant Factor: {interpretation['dominant_factor']}")
        print(f"  Efficiency Ratio: {interpretation['efficiency_ratio']:.4f}")
        print("\nRecommendations:")
        for rec in interpretation['recommendations']:
            print(f"  - {rec}")

if __name__ == "__main__":
    main()