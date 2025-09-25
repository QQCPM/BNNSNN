"""
Spiking Neural Network for Pet Energy - Temporal Uncertainty Demonstration
Shows how timing uncertainty affects neural computation and energy patterns

Key Educational Concepts:
- Temporal uncertainty: "When will the next spike occur?"
- Membrane potential buildup with noise
- Spike timing variability affects energy prediction
- Event-driven vs continuous computation

The pet's energy shows: "Energy bursts are irregular - timing has uncertainty!"
This makes temporal dynamics and biological plausibility tangible.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


class SimpleLIFNeuron:
    """
    Simple Leaky Integrate-and-Fire Neuron demonstrating temporal uncertainty

    Key Equation: V(t+1) = β*V(t) + I(t) + noise
    Spikes when: V(t) > threshold

    Educational Focus:
    - Membrane potential evolution over time
    - Threshold crossing is probabilistic (due to noise)
    - Spike timing has natural variability
    """

    def __init__(self, beta: float = 0.9, threshold: float = 1.0, noise_std: float = 0.05):
        self.beta = beta  # Decay factor (0.9 = 10% decay per time step)
        self.threshold = threshold
        self.noise_std = noise_std
        self.membrane_potential = 0.0
        self.spike_history = []
        self.membrane_history = []

    def step(self, input_current: float) -> Tuple[bool, float]:
        """
        Single time step of LIF neuron

        Returns:
            (spike_occurred, membrane_potential)
        """
        # Add noise for temporal uncertainty
        noise = np.random.normal(0, self.noise_std)

        # Leaky integration with noise
        self.membrane_potential = (
            self.beta * self.membrane_potential +
            input_current +
            noise
        )

        # Check for spike
        spike = self.membrane_potential > self.threshold

        # Reset if spike occurred
        if spike:
            self.membrane_potential = 0.0

        # Store history for visualization
        self.membrane_history.append(self.membrane_potential)
        self.spike_history.append(spike)

        return spike, self.membrane_potential

    def get_spike_pattern_uncertainty(self) -> Dict:
        """
        Analyze uncertainty in spike patterns

        Educational insight: Even with same input, timing varies!
        """
        if len(self.spike_history) < 10:
            return {'insufficient_data': True}

        # Find spike times
        spike_times = [i for i, spike in enumerate(self.spike_history) if spike]

        if len(spike_times) < 2:
            return {'no_spikes': True}

        # Inter-spike intervals
        intervals = np.diff(spike_times)

        return {
            'total_spikes': len(spike_times),
            'mean_interval': np.mean(intervals),
            'interval_std': np.std(intervals),
            'coefficient_of_variation': np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0,
            'spike_times': spike_times,
            'intervals': intervals.tolist()
        }


class PetEnergyTemporalSNN:
    """
    Simple SNN for Pet Energy with Temporal Uncertainty Focus

    Educational Architecture:
    - 3 LIF neurons representing different energy aspects
    - Input: Recent interactions drive current injection
    - Output: Energy level from spike rate
    - Key concept: Same interactions → different spike patterns → energy uncertainty
    """

    def __init__(self):
        # Create 3 LIF neurons with slightly different properties
        self.neurons = [
            SimpleLIFNeuron(beta=0.85, threshold=1.0, noise_std=0.08),  # Fast, noisy
            SimpleLIFNeuron(beta=0.95, threshold=1.2, noise_std=0.05),  # Slow, stable
            SimpleLIFNeuron(beta=0.90, threshold=0.8, noise_std=0.06),  # Medium
        ]

        # Simulation parameters
        self.time_steps = 50  # Short simulation for real-time
        self.current_step = 0

        # Energy calculation
        self.energy_history = []
        self.spike_rate_history = []

    def simulate_energy_response(self, interactions: List[Dict]) -> Dict:
        """
        Simulate energy response to interactions with temporal uncertainty

        Key Educational Demo:
        - Same interactions can produce different energy patterns
        - Uncertainty comes from spike timing variability
        - Shows biological plausibility of neural computation
        """
        # Convert interactions to input currents
        input_currents = self._interactions_to_currents(interactions)

        # Simulate neural response
        all_spikes = []
        all_membranes = []

        for t in range(self.time_steps):
            current = input_currents[t] if t < len(input_currents) else 0.0

            step_spikes = []
            step_membranes = []

            for neuron in self.neurons:
                spike, membrane = neuron.step(current)
                step_spikes.append(spike)
                step_membranes.append(membrane)

            all_spikes.append(step_spikes)
            all_membranes.append(step_membranes)

        # Calculate energy from spike rates
        energy_level = self._spikes_to_energy(all_spikes)

        # Analyze temporal uncertainty
        uncertainty_analysis = self._analyze_temporal_uncertainty()

        return {
            'energy_prediction': energy_level,
            'spike_patterns': all_spikes,
            'membrane_traces': all_membranes,
            'temporal_uncertainty': uncertainty_analysis,
            'simulation_steps': self.time_steps
        }

    def _interactions_to_currents(self, interactions: List[Dict]) -> List[float]:
        """Convert interaction events to input currents"""
        currents = []
        current_time = datetime.now()

        for t in range(self.time_steps):
            # Check for interactions in this time window
            time_window = current_time - timedelta(seconds=self.time_steps - t)
            current = 0.0

            for interaction in interactions:
                interaction_time = interaction.get('timestamp', current_time)
                if abs((interaction_time - time_window).total_seconds()) < 1.0:
                    # Add current based on interaction type
                    interaction_current = {
                        'play': 0.8,
                        'data_feeding': 0.5,
                        'rest': -0.2,
                        'pet_interaction': 0.6
                    }.get(interaction.get('type', 'pet_interaction'), 0.3)

                    current += interaction_current

            currents.append(current)

        return currents

    def _spikes_to_energy(self, all_spikes: List[List[bool]]) -> float:
        """Convert spike patterns to energy level"""
        if not all_spikes:
            return 0.5

        # Count spikes across all neurons and time
        total_spikes = sum(sum(step_spikes) for step_spikes in all_spikes)
        max_possible_spikes = len(self.neurons) * len(all_spikes)

        # Convert spike rate to energy (0-1)
        spike_rate = total_spikes / max_possible_spikes if max_possible_spikes > 0 else 0
        energy = min(1.0, spike_rate * 3.0)  # Scale factor for reasonable energy levels

        self.energy_history.append(energy)
        self.spike_rate_history.append(spike_rate)

        return energy

    def _analyze_temporal_uncertainty(self) -> Dict:
        """
        Analyze temporal uncertainty in spike patterns
        Key educational insight: timing variability affects computation
        """
        uncertainties = []

        for neuron in self.neurons:
            uncertainty = neuron.get_spike_pattern_uncertainty()
            if 'coefficient_of_variation' in uncertainty:
                uncertainties.append(uncertainty['coefficient_of_variation'])

        if not uncertainties:
            return {'no_analysis': True}

        mean_uncertainty = np.mean(uncertainties)

        # Classify uncertainty level
        if mean_uncertainty < 0.3:
            uncertainty_level = "Low temporal uncertainty - regular spike patterns"
        elif mean_uncertainty < 0.7:
            uncertainty_level = "Moderate temporal uncertainty - some variability"
        else:
            uncertainty_level = "High temporal uncertainty - irregular spikes"

        return {
            'mean_coefficient_of_variation': mean_uncertainty,
            'uncertainty_level': uncertainty_level,
            'individual_neurons': [n.get_spike_pattern_uncertainty() for n in self.neurons],
            'interpretation': self._interpret_temporal_uncertainty(mean_uncertainty)
        }

    def _interpret_temporal_uncertainty(self, uncertainty: float) -> List[str]:
        """Generate educational explanations about temporal uncertainty"""
        interpretations = []

        if uncertainty < 0.3:
            interpretations.append("🎯 Neurons are firing quite regularly - predictable energy patterns")
            interpretations.append("⚡ Low temporal uncertainty means reliable energy computation")
        elif uncertainty < 0.7:
            interpretations.append("🎲 Moderate spike timing variability - some energy uncertainty")
            interpretations.append("🧠 This is typical biological neural variability")
        else:
            interpretations.append("🌊 High temporal uncertainty - energy patterns are quite variable")
            interpretations.append("🔬 Irregular spiking shows how noise affects neural computation")

        interpretations.append(f"📊 Timing variability: {uncertainty:.3f} (0=regular, >1=very irregular)")

        return interpretations

    def get_educational_insights(self) -> Dict:
        """Educational insights about temporal uncertainty in neural computation"""
        return {
            'key_concepts': [
                "⏰ Spike timing has natural variability due to biological noise",
                "🧮 Same input can produce different spike patterns",
                "📈 Energy uncertainty comes from temporal dynamics",
                "🧠 This is how real neurons work - not perfectly precise!"
            ],
            'snn_concepts': {
                'temporal_coding': "Information encoded in spike timing, not just rates",
                'membrane_dynamics': "Voltage builds up and decays over time",
                'threshold_crossing': "Probabilistic due to noise and dynamics",
                'event_driven': "Computation happens only when spikes occur"
            },
            'uncertainty_sources': [
                "Membrane potential noise",
                "Threshold variability",
                "Input timing jitter",
                "Synaptic transmission delays"
            ]
        }

    def visualize_temporal_patterns(self) -> Dict:
        """
        Prepare data for visualizing temporal uncertainty
        Shows membrane traces and spike patterns
        """
        if not self.neurons or not self.neurons[0].membrane_history:
            return {'no_data': True}

        visualization_data = {
            'time_steps': list(range(len(self.neurons[0].membrane_history))),
            'membrane_traces': [neuron.membrane_history for neuron in self.neurons],
            'spike_patterns': [neuron.spike_history for neuron in self.neurons],
            'thresholds': [neuron.threshold for neuron in self.neurons],
            'neuron_labels': ['Fast & Noisy', 'Slow & Stable', 'Medium'],
            'energy_history': self.energy_history[-20:],  # Last 20 values
            'spike_rate_history': self.spike_rate_history[-20:]
        }

        return visualization_data


def create_temporal_energy_snn() -> PetEnergyTemporalSNN:
    """Create and test temporal energy SNN"""
    snn = PetEnergyTemporalSNN()

    # Test simulation
    test_interactions = [
        {
            'type': 'play',
            'timestamp': datetime.now() - timedelta(seconds=2)
        },
        {
            'type': 'data_feeding',
            'timestamp': datetime.now() - timedelta(seconds=1)
        }
    ]

    result = snn.simulate_energy_response(test_interactions)

    print(f"✅ Temporal Energy SNN initialized")
    print(f"⚡ Test energy: {result['energy_prediction']:.3f}")
    print(f"🧠 Temporal uncertainty: {result['temporal_uncertainty'].get('uncertainty_level', 'N/A')}")

    return snn


# Educational demonstration
if __name__ == "__main__":
    print("⚡ Spiking Neural Network Demo - Temporal Uncertainty in Pet Energy")

    snn = create_temporal_energy_snn()

    # Show how same interactions can produce different results
    print("\n🎲 Demonstrating temporal uncertainty:")
    print("Same interactions, different spike patterns due to noise:")

    interactions = [{'type': 'play', 'timestamp': datetime.now()}]

    for run in range(3):
        result = snn.simulate_energy_response(interactions)
        energy = result['energy_prediction']
        uncertainty_info = result['temporal_uncertainty']

        print(f"\nRun {run+1}: Energy = {energy:.3f}")
        if 'uncertainty_level' in uncertainty_info:
            print(f"  {uncertainty_info['uncertainty_level']}")

    # Show educational insights
    insights = snn.get_educational_insights()
    print(f"\n🔬 Key SNN Concepts Demonstrated:")
    for concept in insights['key_concepts']:
        print(f"  {concept}")

    print("\n✅ Temporal SNN demonstration complete!")