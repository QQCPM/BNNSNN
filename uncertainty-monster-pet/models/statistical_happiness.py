"""
Statistical Analysis of Pet Happiness - Distribution Uncertainty Demonstration
Uses basic moment analysis to quantify uncertainty in happiness patterns

Key Educational Concepts:
- Statistical moments tell us about distribution shape
- Mean (1st moment): Central tendency
- Variance (2nd moment): Spread/uncertainty
- Skewness (3rd moment): Asymmetry
- How distribution shape reveals happiness patterns

The pet's happiness shows: "Your pet is usually 70% happy, but varies ±20%"
This makes statistical concepts tangible and meaningful.
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import scipy.stats as stats


class PetHappinessStatistics:
    """
    Simple statistical analysis of pet happiness focusing on uncertainty

    Educational Focus:
    - Moments as measures of uncertainty
    - How distribution shape tells a story
    - Central tendency vs variability
    - Statistical confidence in happiness assessment
    """

    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.happiness_values = []
        self.timestamps = []
        self.moments_history = []

    def add_happiness_value(self, happiness: float, timestamp: Optional[datetime] = None) -> None:
        """
        Add new happiness observation

        Args:
            happiness: Happiness value (0.0 - 1.0)
            timestamp: When observed (defaults to now)
        """
        if not (0.0 <= happiness <= 1.0):
            raise ValueError("Happiness must be between 0.0 and 1.0")

        self.happiness_values.append(happiness)
        self.timestamps.append(timestamp or datetime.now())

        # Maintain history limit
        if len(self.happiness_values) > self.max_history:
            self.happiness_values.pop(0)
            self.timestamps.pop(0)

    def compute_basic_moments(self) -> Dict:
        """
        Compute basic statistical moments

        Educational Focus: What do these numbers tell us about happiness?

        Returns:
            Dictionary with moments and their interpretations
        """
        if len(self.happiness_values) < 3:
            return {'insufficient_data': True, 'need_more': 3 - len(self.happiness_values)}

        values = np.array(self.happiness_values)

        # First moment: Mean (central tendency)
        mean = np.mean(values)

        # Second moment: Variance (spread/uncertainty)
        variance = np.var(values, ddof=1) if len(values) > 1 else 0
        std_dev = np.sqrt(variance)

        # Third moment: Skewness (asymmetry)
        skewness = stats.skew(values) if len(values) > 2 else 0

        # Fourth moment: Kurtosis (tail heaviness)
        kurtosis = stats.kurtosis(values) if len(values) > 3 else 0

        moments = {
            'sample_size': len(values),
            'mean': mean,
            'variance': variance,
            'std_deviation': std_dev,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'min_value': np.min(values),
            'max_value': np.max(values),
            'range': np.max(values) - np.min(values)
        }

        # Add interpretations
        moments['interpretations'] = self._interpret_moments(moments)

        # Store in history
        self.moments_history.append({
            'timestamp': datetime.now(),
            'moments': moments
        })

        return moments

    def _interpret_moments(self, moments: Dict) -> List[str]:
        """
        Generate educational interpretations of statistical moments

        Makes abstract statistics concrete and meaningful
        """
        interpretations = []

        # Mean interpretation
        mean = moments['mean']
        if mean > 0.8:
            interpretations.append(f"😍 Very happy pet! Average happiness is {mean:.2f}")
        elif mean > 0.6:
            interpretations.append(f"😊 Generally happy pet with {mean:.2f} average")
        elif mean > 0.4:
            interpretations.append(f"😐 Moderate happiness level at {mean:.2f}")
        else:
            interpretations.append(f"😢 Pet needs attention - only {mean:.2f} average happiness")

        # Variance interpretation (uncertainty measure!)
        std_dev = moments['std_deviation']
        if std_dev > 0.25:
            interpretations.append(f"🎢 Very variable moods! Happiness varies by ±{std_dev:.2f}")
        elif std_dev > 0.15:
            interpretations.append(f"🎭 Moderate mood swings, varies ±{std_dev:.2f}")
        elif std_dev > 0.05:
            interpretations.append(f"📏 Fairly stable moods, varies ±{std_dev:.2f}")
        else:
            interpretations.append(f"🔧 Very consistent happiness! Only ±{std_dev:.2f} variation")

        # Skewness interpretation
        skewness = moments['skewness']
        if abs(skewness) > 0.5:
            if skewness > 0:
                interpretations.append("📈 Mostly moderate with occasional high happiness spikes!")
            else:
                interpretations.append("📉 Generally happy with occasional low periods")
        else:
            interpretations.append("⚖️ Balanced happiness distribution - symmetric moods")

        return interpretations

    def calculate_confidence_intervals(self, confidence_level: float = 0.95) -> Dict:
        """
        Calculate confidence intervals for happiness

        Educational concept: How sure are we about the mean happiness?
        """
        if len(self.happiness_values) < 2:
            return {'insufficient_data': True}

        values = np.array(self.happiness_values)
        mean = np.mean(values)
        std_error = stats.sem(values)  # Standard error of the mean

        # Confidence interval
        confidence_interval = stats.t.interval(
            confidence_level,
            len(values) - 1,
            loc=mean,
            scale=std_error
        )

        return {
            'mean': mean,
            'standard_error': std_error,
            'confidence_level': confidence_level,
            'confidence_interval': confidence_interval,
            'interpretation': f"We're {confidence_level*100:.0f}% confident true happiness is between {confidence_interval[0]:.3f} and {confidence_interval[1]:.3f}"
        }

    def detect_happiness_patterns(self) -> Dict:
        """
        Detect patterns in happiness data

        Educational Focus: What stories do the statistics tell?
        """
        if len(self.happiness_values) < 10:
            return {'insufficient_data': True}

        patterns = {}

        # Recent vs older happiness
        mid_point = len(self.happiness_values) // 2
        recent_values = self.happiness_values[mid_point:]
        older_values = self.happiness_values[:mid_point]

        recent_mean = np.mean(recent_values)
        older_mean = np.mean(older_values)

        patterns['trend'] = {
            'recent_mean': recent_mean,
            'older_mean': older_mean,
            'change': recent_mean - older_mean,
            'direction': 'improving' if recent_mean > older_mean else 'declining' if recent_mean < older_mean else 'stable'
        }

        # Stability analysis
        recent_std = np.std(recent_values)
        older_std = np.std(older_values)

        patterns['stability'] = {
            'recent_variability': recent_std,
            'older_variability': older_std,
            'stability_change': 'more_stable' if recent_std < older_std else 'less_stable' if recent_std > older_std else 'same'
        }

        # Time-based patterns (if we have timestamps)
        if len(self.timestamps) == len(self.happiness_values):
            patterns['temporal'] = self._analyze_temporal_patterns()

        return patterns

    def _analyze_temporal_patterns(self) -> Dict:
        """Simple time-based pattern analysis"""
        # Group by hour of day
        hourly_happiness = {}

        for happiness, timestamp in zip(self.happiness_values, self.timestamps):
            hour = timestamp.hour
            if hour not in hourly_happiness:
                hourly_happiness[hour] = []
            hourly_happiness[hour].append(happiness)

        # Find best and worst hours
        hourly_means = {hour: np.mean(values) for hour, values in hourly_happiness.items() if len(values) >= 2}

        if hourly_means:
            best_hour = max(hourly_means, key=hourly_means.get)
            worst_hour = min(hourly_means, key=hourly_means.get)

            return {
                'best_hour': best_hour,
                'worst_hour': worst_hour,
                'best_happiness': hourly_means[best_hour],
                'worst_happiness': hourly_means[worst_hour],
                'hourly_pattern': hourly_means
            }

        return {'insufficient_temporal_data': True}

    def get_uncertainty_summary(self) -> Dict:
        """
        Summarize uncertainty in happiness assessment

        Key educational point: Statistics quantify our uncertainty!
        """
        if len(self.happiness_values) < 3:
            return {
                'status': 'insufficient_data',
                'message': 'Need more happiness observations to assess uncertainty'
            }

        moments = self.compute_basic_moments()
        confidence = self.calculate_confidence_intervals()

        # Overall uncertainty assessment
        std_dev = moments['std_deviation']
        sample_size = moments['sample_size']

        if std_dev < 0.1 and sample_size > 20:
            uncertainty_level = "Low"
            message = "Very confident in happiness assessment"
        elif std_dev < 0.2 and sample_size > 10:
            uncertainty_level = "Moderate"
            message = "Reasonably confident in happiness patterns"
        else:
            uncertainty_level = "High"
            message = "High uncertainty - need more data or pet has very variable moods"

        return {
            'uncertainty_level': uncertainty_level,
            'message': message,
            'key_metrics': {
                'mean_happiness': moments['mean'],
                'happiness_variability': std_dev,
                'sample_size': sample_size,
                'confidence_interval_width': confidence['confidence_interval'][1] - confidence['confidence_interval'][0]
            },
            'recommendations': self._generate_uncertainty_recommendations(uncertainty_level, std_dev, sample_size)
        }

    def _generate_uncertainty_recommendations(self, uncertainty_level: str, std_dev: float, sample_size: int) -> List[str]:
        """Generate recommendations based on uncertainty analysis"""
        recommendations = []

        if uncertainty_level == "High":
            if sample_size < 20:
                recommendations.append("📊 Collect more happiness data for better certainty")
            if std_dev > 0.25:
                recommendations.append("🎭 Your pet has very variable moods - this is the nature of their personality")

        if sample_size < 10:
            recommendations.append("⏰ More interactions needed for reliable statistical analysis")

        if uncertainty_level == "Low":
            recommendations.append("✅ Happiness patterns are well established - statistics are reliable")

        return recommendations

    def get_educational_insights(self) -> Dict:
        """Educational insights about statistical concepts demonstrated"""
        return {
            'statistical_concepts': [
                "📊 Mean tells us the typical happiness level",
                "📏 Standard deviation measures mood variability (uncertainty!)",
                "📈 Skewness shows if happiness is asymmetric",
                "🎯 Confidence intervals quantify our uncertainty about the true mean"
            ],
            'uncertainty_concepts': {
                'variability': "High standard deviation = high uncertainty in individual predictions",
                'confidence_intervals': "Range where we expect the true mean to lie",
                'sample_size_effect': "More data = smaller confidence intervals = less uncertainty",
                'distribution_shape': "Skewness and kurtosis tell us about rare events"
            },
            'practical_insights': [
                "Statistics help us understand pet personality patterns",
                "Uncertainty quantification is crucial for reliable conclusions",
                "Distribution shape reveals more than just averages",
                "Larger samples give more confident statistical conclusions"
            ]
        }


def create_happiness_statistics() -> PetHappinessStatistics:
    """Create and test happiness statistics analyzer"""
    analyzer = PetHappinessStatistics()

    # Add some test data to demonstrate
    np.random.seed(42)  # For reproducible demo
    test_happiness = np.random.beta(2, 1.5, 20)  # Beta distribution for realistic happiness

    for i, happiness in enumerate(test_happiness):
        analyzer.add_happiness_value(
            happiness,
            datetime.now() - timedelta(hours=20-i)
        )

    moments = analyzer.compute_basic_moments()

    print(f"✅ Happiness Statistics Analyzer initialized")
    print(f"📊 Test analysis - Mean: {moments['mean']:.3f}, Std: {moments['std_deviation']:.3f}")
    print(f"🎭 {moments['interpretations'][0]}")

    return analyzer


# Educational demonstration
if __name__ == "__main__":
    print("📊 Statistical Analysis Demo - Happiness Distribution Uncertainty")

    analyzer = create_happiness_statistics()

    # Demonstrate statistical concepts
    print("\n📚 Key Statistical Concepts:")

    # Show moments
    moments = analyzer.compute_basic_moments()
    for interpretation in moments['interpretations'][:2]:
        print(f"  {interpretation}")

    # Show confidence intervals
    confidence = analyzer.calculate_confidence_intervals()
    print(f"  🎯 {confidence['interpretation']}")

    # Show uncertainty summary
    uncertainty = analyzer.get_uncertainty_summary()
    print(f"  ⚖️ Uncertainty level: {uncertainty['uncertainty_level']}")
    print(f"  💭 {uncertainty['message']}")

    # Show educational insights
    insights = analyzer.get_educational_insights()
    print(f"\n🔬 Key Statistical Concepts Demonstrated:")
    for concept in insights['statistical_concepts']:
        print(f"  {concept}")

    print("\n✅ Statistical analysis demonstration complete!")