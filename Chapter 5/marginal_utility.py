from dataclasses import dataclass
from typing import Dict
import numpy as np
from enum import Enum

class DecisionCategory(Enum):
    """Categories for data collection decisions based on marginal utility analysis"""
    IMPROVE_QUALITY = "Improve Quality"  # When quality improvement gives highest returns
    INCREASE_VOLUME = "Increase Volume"  # When volume increase is most beneficial
    MAINTAIN_CURRENT = "Maintain Current"  # When current state is optimal
    STOP_COLLECTION = "Stop Collection"  # When marginal utilities are below thresholds

@dataclass
class MarginalUtilityConfig:
    """Configuration parameters for marginal utility calculation
    
    Attributes:
        alpha: Weight for quality component (0-1)
        beta: Weight for volume component (0-1)
        gamma: Weight for cost component (0-1)
        quality_threshold: Minimum acceptable quality marginal utility
        volume_threshold: Minimum acceptable volume marginal utility
    """
    alpha: float  # Quality weight
    beta: float   # Volume weight
    gamma: float  # Cost weight
    quality_threshold: float = 0.1  # Default quality threshold
    volume_threshold: float = 0.1   # Default volume threshold
    
    def __post_init__(self):
        """Validate that weights sum to 1"""
        if not np.isclose(self.alpha + self.beta + self.gamma, 1.0):
            raise ValueError("Weights must sum to 1")

class EnhancedUtilityCalculator:
    """Calculator for marginal utility analysis of data collection"""
    
    def __init__(self, config: MarginalUtilityConfig):
        """Initialize calculator with configuration parameters"""
        self.config = config

    def quality_benefit_derivative(self, q: float) -> float:
        """Calculate partial derivative of benefit with respect to quality
        
        Args:
            q: Current quality level (0-1)
        Returns:
            Marginal benefit of quality improvement with diminishing returns
        """
        # Exponential diminishing returns for quality
        return np.exp(-2 * q)  # Higher quality = lower marginal benefit

    def volume_benefit_derivative(self, v: int) -> float:
        """Calculate partial derivative of benefit with respect to volume
        
        Args:
            v: Current data volume
        Returns:
            Marginal benefit of volume increase with diminishing returns
        """
        # Logarithmic diminishing returns for volume
        return 1000.0 / (v * np.log(v + 100))

    def cost_derivative_quality(self, q: float, v: int) -> float:
        """Calculate partial derivative of cost with respect to quality
        
        Args:
            q: Quality level
            v: Volume
        Returns:
            Marginal cost of quality improvement
        """
        # Exponential cost increase with quality and volume
        return 0.1 * v * np.exp(2 * q) / 1e4

    def cost_derivative_volume(self, q: float, v: int) -> float:
        """Calculate partial derivative of cost with respect to volume
        
        Args:
            q: Quality level
            v: Volume
        Returns:
            Marginal cost of volume increase
        """
        # Linear cost increase with quality, logarithmic with volume
        return 0.2 * q * np.log(v + 100) / 1e3

    def marginal_quality_utility(self, q: float, v: int) -> float:
        """Calculate marginal utility for quality improvement
        
        Combines benefit and cost derivatives weighted by configuration parameters
        """
        return (self.config.alpha * self.quality_benefit_derivative(q) - 
                self.config.gamma * self.cost_derivative_quality(q, v))

    def marginal_volume_utility(self, q: float, v: int) -> float:
        """Calculate marginal utility for volume increase
        
        Combines benefit and cost derivatives weighted by configuration parameters
        """
        return (self.config.beta * self.volume_benefit_derivative(v) - 
                self.config.gamma * self.cost_derivative_volume(q, v))

    def analyze_marginal_utilities(self, q: float, v: int) -> Dict:
        """Analyze marginal utilities and recommend action
        
        Args:
            q: Current quality level
            v: Current volume
            
        Returns:
            Dictionary containing marginal utilities and recommended decision
        """
        # Calculate marginal utilities
        mu_q = self.marginal_quality_utility(q, v)
        mu_v = self.marginal_volume_utility(q, v)
        
        # Decision logic based on marginal utilities and thresholds
        if mu_q < self.config.quality_threshold and mu_v < self.config.volume_threshold:
            decision = DecisionCategory.STOP_COLLECTION  # Both utilities below threshold
        elif mu_q > mu_v and mu_q > self.config.quality_threshold:
            decision = DecisionCategory.IMPROVE_QUALITY  # Quality improvement most beneficial
        elif mu_v > self.config.volume_threshold:
            decision = DecisionCategory.INCREASE_VOLUME  # Volume increase most beneficial
        else:
            decision = DecisionCategory.MAINTAIN_CURRENT  # Current state is optimal

        return {
            'quality_mu': mu_q,
            'volume_mu': mu_v,
            'decision': decision
        }

def simulate_marginal_utility():
    """Run simulation with realistic scenarios to demonstrate utility analysis"""
    # Initialize with realistic weights
    config = MarginalUtilityConfig(
        alpha=0.4,    # Weight for quality
        beta=0.4,     # Weight for volume
        gamma=0.2,    # Weight for cost
        quality_threshold=0.1,
        volume_threshold=0.1
    )
    
    calculator = EnhancedUtilityCalculator(config)
    
    # Test scenarios with realistic combinations
    scenarios = [
        {'quality': 0.2, 'volume': 1000},    # Low quality, low volume
        {'quality': 0.5, 'volume': 5000},    # Medium quality, medium volume
        {'quality': 0.8, 'volume': 10000},   # High quality, medium volume
        {'quality': 0.3, 'volume': 50000},   # Low quality, high volume
        {'quality': 0.9, 'volume': 100000},  # High quality, high volume
        {'quality': 0.4, 'volume': 25000},   # Medium-low quality, medium-high volume
        {'quality': 0.7, 'volume': 75000},   # Medium-high quality, high volume
        {'quality': 0.6, 'volume': 15000}    # Medium quality, medium-high volume
    ]
    
    # Run and print results with detailed analysis
    print("\nMarginal Utility Simulation Results:")
    print("=" * 60)
    
    for i, scenario in enumerate(scenarios, 1):
        result = calculator.analyze_marginal_utilities(scenario['quality'], 
                                                    scenario['volume'])
        
        print(f"\nScenario {i}:")
        print(f"Quality Level: {scenario['quality']:.2f}")
        print(f"Data Volume: {scenario['volume']:,}")
        print("-" * 40)
        print(f"Quality Marginal Utility: {result['quality_mu']:.4f}")
        print(f"Volume Marginal Utility: {result['volume_mu']:.4f}")
        print(f"Recommended Action: {result['decision'].value}")
        
        # Add interpretation
        if result['decision'] == DecisionCategory.IMPROVE_QUALITY:
            print("→ Focus on improving data quality")
        elif result['decision'] == DecisionCategory.INCREASE_VOLUME:
            print("→ Focus on collecting more data")
        elif result['decision'] == DecisionCategory.MAINTAIN_CURRENT:
            print("→ Current state is optimal")
        else:
            print("→ Consider stopping data collection")

if __name__ == "__main__":
    simulate_marginal_utility()