"""
Uncertainty Monster Pet - Educational Demo
Demonstrates three different approaches to handling uncertainty in AI

🧠 Bayesian Neural Network: "I'm 70% sure your pet is 80% happy"
⚡ Spiking Neural Network: "Energy patterns have timing uncertainty"
📊 Statistical Analysis: "Happiness distribution shows mood variability"

This is the SIMPLIFIED, EDUCATIONAL version focusing on core uncertainty concepts.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

# Import our uncertainty-focused models
from models.bayesian_mood import create_bayesian_mood_predictor
from models.temporal_energy import create_temporal_energy_snn
from models.statistical_happiness import create_happiness_statistics


def main():
    """Main educational interface for uncertainty concepts"""

    st.set_page_config(
        page_title="🎯 Uncertainty Monster Pet",
        page_icon="🎯",
        layout="wide"
    )

    st.markdown("""
        # 🎯 Uncertainty Monster Pet
        ### Learn how AI handles uncertainty through three different approaches!

        **🧠 Bayesian NN**: Predictions with confidence bounds
        **⚡ Spiking NN**: Temporal patterns with timing variability
        **📊 Statistics**: Distribution analysis with moment calculations
    """)

    # Initialize models in session state
    if 'bayesian_model' not in st.session_state:
        with st.spinner("🧠 Loading Bayesian Neural Network..."):
            st.session_state.bayesian_model = create_bayesian_mood_predictor()

    if 'spiking_model' not in st.session_state:
        with st.spinner("⚡ Loading Spiking Neural Network..."):
            st.session_state.spiking_model = create_temporal_energy_snn()

    if 'stats_model' not in st.session_state:
        with st.spinner("📊 Loading Statistical Analyzer..."):
            st.session_state.stats_model = create_happiness_statistics()

    # Initialize pet state
    if 'pet_interactions' not in st.session_state:
        st.session_state.pet_interactions = []
        st.session_state.interaction_count = 0
        st.session_state.current_mood = 0.5
        st.session_state.current_energy = 0.7

    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎮 Interact with Pet",
        "🧠 Bayesian Uncertainty",
        "⚡ Temporal Uncertainty",
        "📊 Statistical Uncertainty"
    ])

    with tab1:
        display_pet_interaction()

    with tab2:
        display_bayesian_analysis()

    with tab3:
        display_temporal_analysis()

    with tab4:
        display_statistical_analysis()


def display_pet_interaction():
    """Simple pet interaction interface"""
    st.subheader("🐾 Your Uncertainty Pet")

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        # Simple pet visualization
        mood = st.session_state.current_mood
        energy = st.session_state.current_energy

        # Choose pet emoji based on mood and energy
        if mood > 0.8 and energy > 0.7:
            pet_emoji = "🤩"  # Very happy and energetic
        elif mood > 0.6:
            pet_emoji = "😊"  # Happy
        elif mood > 0.4:
            pet_emoji = "😐"  # Neutral
        else:
            pet_emoji = "😔"  # Sad

        st.markdown(f"""
            <div style="text-align: center; padding: 2rem; border: 2px solid #667eea; border-radius: 15px; background: #f8f9ff;">
                <div style="font-size: 4rem;">{pet_emoji}</div>
                <h3>Uncertainty</h3>
                <p><strong>Mood:</strong> {mood:.3f} ({'😍' * int(mood * 5)})</p>
                <p><strong>Energy:</strong> {energy:.3f} ({'⚡' * int(energy * 5)})</p>
                <p><strong>Interactions:</strong> {st.session_state.interaction_count}</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("🎮 Pet Interactions")

        # Interaction buttons
        col2a, col2b = st.columns(2)

        with col2a:
            if st.button("🎾 Play", type="primary"):
                interact_with_pet("play")
                st.rerun()

            if st.button("🍽️ Feed Data"):
                interact_with_pet("data_feeding")
                st.rerun()

        with col2b:
            if st.button("💤 Rest"):
                interact_with_pet("rest")
                st.rerun()

            if st.button("🔄 Reset Pet"):
                reset_pet()
                st.rerun()

    with col3:
        st.subheader("📊 Quick Stats")
        st.metric("Current Mood", f"{st.session_state.current_mood:.3f}")
        st.metric("Current Energy", f"{st.session_state.current_energy:.3f}")
        st.metric("Total Interactions", st.session_state.interaction_count)


def interact_with_pet(interaction_type: str):
    """Process pet interaction and update all uncertainty models"""

    # Record interaction
    interaction = {
        'type': interaction_type,
        'timestamp': datetime.now()
    }
    st.session_state.pet_interactions.append(interaction)
    st.session_state.interaction_count += 1

    # Simple mood/energy updates
    interaction_effects = {
        'play': {'mood': 0.1, 'energy': -0.05},
        'data_feeding': {'mood': 0.15, 'energy': 0.1},
        'rest': {'mood': 0.05, 'energy': 0.2}
    }

    effect = interaction_effects.get(interaction_type, {'mood': 0.05, 'energy': 0.0})

    st.session_state.current_mood = np.clip(
        st.session_state.current_mood + effect['mood'] + np.random.normal(0, 0.05),
        0.0, 1.0
    )
    st.session_state.current_energy = np.clip(
        st.session_state.current_energy + effect['energy'] + np.random.normal(0, 0.03),
        0.0, 1.0
    )

    # Update statistical model
    st.session_state.stats_model.add_happiness_value(st.session_state.current_mood)

    st.success(f"✅ {interaction_type.replace('_', ' ').title()} completed!")


def reset_pet():
    """Reset pet to initial state"""
    st.session_state.pet_interactions = []
    st.session_state.interaction_count = 0
    st.session_state.current_mood = 0.5
    st.session_state.current_energy = 0.7

    # Reset models
    st.session_state.stats_model = create_happiness_statistics()

    st.success("🔄 Pet has been reset!")


def display_bayesian_analysis():
    """Show Bayesian neural network uncertainty analysis"""
    st.subheader("🧠 Bayesian Neural Network - Prediction Uncertainty")

    st.markdown("""
        **Key Concept**: Neural network weights are probability distributions, not fixed numbers.
        This lets us quantify how confident the network is in its predictions!
    """)

    if st.session_state.interaction_count == 0:
        st.info("🎮 Interact with your pet first to see Bayesian uncertainty in action!")
        return

    # Get Bayesian prediction
    current_time = datetime.now().hour / 24.0

    prediction = st.session_state.bayesian_model.predict_with_uncertainty(
        time_of_day=current_time,
        interaction_count=st.session_state.interaction_count,
        previous_mood=st.session_state.current_mood
    )

    col1, col2 = st.columns(2)

    with col1:
        # Prediction with uncertainty
        st.subheader("🎯 Mood Prediction")

        mood_pred = prediction['mood_prediction']
        uncertainty = prediction['uncertainty_std']
        confidence = prediction['confidence']

        st.metric("Predicted Mood", f"{mood_pred:.3f}")
        st.metric("Uncertainty (±)", f"{uncertainty:.3f}")
        st.metric("Confidence", f"{confidence:.3f}")

        # Confidence interval visualization
        lower, upper = prediction['confidence_interval_95']

        fig_pred = go.Figure()

        # Add prediction point
        fig_pred.add_trace(go.Scatter(
            x=[0], y=[mood_pred],
            mode='markers',
            marker=dict(size=15, color='blue'),
            name='Prediction'
        ))

        # Add uncertainty bars
        fig_pred.add_trace(go.Scatter(
            x=[0, 0], y=[lower, upper],
            mode='lines',
            line=dict(color='red', width=5),
            name='95% Confidence'
        ))

        fig_pred.update_layout(
            title="Bayesian Prediction with Uncertainty",
            xaxis_title="",
            yaxis_title="Predicted Mood",
            yaxis=dict(range=[0, 1]),
            showlegend=True,
            height=300
        )

        st.plotly_chart(fig_pred, use_container_width=True)

    with col2:
        # Educational explanations
        st.subheader("🧠 Understanding Bayesian Uncertainty")

        explanations = st.session_state.bayesian_model.explain_uncertainty(prediction)

        for explanation in explanations:
            st.write(f"• {explanation}")

        # Show how uncertainty evolves
        evolution = st.session_state.bayesian_model.get_uncertainty_evolution()

        if not evolution.get('insufficient_data', False):
            fig_evolution = go.Figure()

            fig_evolution.add_trace(go.Scatter(
                x=evolution['interaction_counts'],
                y=evolution['uncertainties'],
                mode='lines+markers',
                name='Uncertainty',
                line=dict(color='red')
            ))

            fig_evolution.update_layout(
                title="Uncertainty vs Interactions",
                xaxis_title="Number of Interactions",
                yaxis_title="Prediction Uncertainty",
                height=250
            )

            st.plotly_chart(fig_evolution, use_container_width=True)

    # Educational insights
    st.subheader("📚 Bayesian Concepts Explained")
    insights = st.session_state.bayesian_model.get_bayesian_insights()

    for concept in insights['concept_demonstrations'][:2]:
        st.write(f"• {concept}")


def display_temporal_analysis():
    """Show spiking neural network temporal uncertainty"""
    st.subheader("⚡ Spiking Neural Network - Temporal Uncertainty")

    st.markdown("""
        **Key Concept**: Neural spikes have timing variability due to biological noise.
        Same inputs can produce different spike patterns, leading to energy uncertainty!
    """)

    if len(st.session_state.pet_interactions) == 0:
        st.info("🎮 Interact with your pet first to see temporal spike patterns!")
        return

    # Simulate SNN response
    recent_interactions = st.session_state.pet_interactions[-5:]  # Last 5 interactions

    snn_result = st.session_state.spiking_model.simulate_energy_response(recent_interactions)

    col1, col2 = st.columns(2)

    with col1:
        # Energy prediction
        st.subheader("⚡ Energy from Spike Patterns")

        energy_pred = snn_result['energy_prediction']
        st.metric("Predicted Energy", f"{energy_pred:.3f}")

        temporal_uncertainty = snn_result['temporal_uncertainty']
        if 'uncertainty_level' in temporal_uncertainty:
            st.write(f"**Temporal Pattern**: {temporal_uncertainty['uncertainty_level']}")

        # Spike pattern visualization
        spike_patterns = np.array(snn_result['spike_patterns'])

        if spike_patterns.size > 0:
            fig_spikes = go.Figure()

            # Show spikes for each neuron
            neuron_names = ['Fast & Noisy', 'Slow & Stable', 'Medium']

            for neuron_idx in range(min(3, spike_patterns.shape[1])):
                neuron_spikes = spike_patterns[:, neuron_idx]
                spike_times = [t for t, spike in enumerate(neuron_spikes) if spike]

                if spike_times:
                    fig_spikes.add_trace(go.Scatter(
                        x=spike_times,
                        y=[neuron_idx] * len(spike_times),
                        mode='markers',
                        marker=dict(size=8, symbol='line-ns'),
                        name=neuron_names[neuron_idx]
                    ))

            fig_spikes.update_layout(
                title="Spike Raster Plot - Temporal Patterns",
                xaxis_title="Time Steps",
                yaxis_title="Neuron",
                yaxis=dict(tickvals=[0, 1, 2], ticktext=neuron_names),
                height=300
            )

            st.plotly_chart(fig_spikes, use_container_width=True)

    with col2:
        # Temporal uncertainty analysis
        st.subheader("🧠 Understanding Temporal Uncertainty")

        if 'interpretation' in temporal_uncertainty:
            for interpretation in temporal_uncertainty['interpretation'][:2]:
                st.write(f"• {interpretation}")

        # Show membrane potential traces
        membrane_traces = np.array(snn_result['membrane_traces'])

        if membrane_traces.size > 0:
            fig_membrane = go.Figure()

            time_steps = list(range(len(membrane_traces)))
            neuron_names = ['Fast & Noisy', 'Slow & Stable', 'Medium']

            for neuron_idx in range(min(3, membrane_traces.shape[1])):
                membrane_trace = membrane_traces[:, neuron_idx]

                fig_membrane.add_trace(go.Scatter(
                    x=time_steps,
                    y=membrane_trace,
                    mode='lines',
                    name=f'{neuron_names[neuron_idx]} Membrane'
                ))

            # Add threshold line
            fig_membrane.add_hline(y=1.0, line_dash="dash", line_color="red",
                                 annotation_text="Spike Threshold")

            fig_membrane.update_layout(
                title="Membrane Potential Over Time",
                xaxis_title="Time Steps",
                yaxis_title="Membrane Potential",
                height=250
            )

            st.plotly_chart(fig_membrane, use_container_width=True)

    # Educational insights
    st.subheader("📚 Spiking Neural Network Concepts")
    insights = st.session_state.spiking_model.get_educational_insights()

    for concept in insights['key_concepts'][:2]:
        st.write(f"• {concept}")


def display_statistical_analysis():
    """Show statistical distribution analysis"""
    st.subheader("📊 Statistical Analysis - Distribution Uncertainty")

    st.markdown("""
        **Key Concept**: Statistical moments tell us about distribution shape and uncertainty.
        Mean, variance, and skewness reveal patterns in your pet's happiness!
    """)

    if st.session_state.interaction_count < 3:
        st.info("🎮 Need at least 3 interactions to perform statistical analysis!")
        return

    # Get statistical analysis
    moments = st.session_state.stats_model.compute_basic_moments()

    if moments.get('insufficient_data', False):
        st.warning("Need more data for statistical analysis")
        return

    col1, col2 = st.columns(2)

    with col1:
        # Statistical moments
        st.subheader("📊 Statistical Moments")

        st.metric("Mean Happiness", f"{moments['mean']:.3f}")
        st.metric("Standard Deviation", f"{moments['std_deviation']:.3f}")
        st.metric("Skewness", f"{moments['skewness']:.3f}")
        st.metric("Sample Size", moments['sample_size'])

        # Happiness distribution
        happiness_values = st.session_state.stats_model.happiness_values

        if len(happiness_values) >= 5:
            fig_hist = go.Figure()

            fig_hist.add_trace(go.Histogram(
                x=happiness_values,
                nbinsx=10,
                name='Happiness Distribution',
                opacity=0.7
            ))

            # Add mean line
            fig_hist.add_vline(
                x=moments['mean'],
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {moments['mean']:.3f}"
            )

            fig_hist.update_layout(
                title="Happiness Distribution",
                xaxis_title="Happiness Level",
                yaxis_title="Frequency",
                height=300
            )

            st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        # Statistical interpretations
        st.subheader("🧠 Understanding Statistical Uncertainty")

        for interpretation in moments['interpretations'][:2]:
            st.write(f"• {interpretation}")

        # Confidence intervals
        confidence = st.session_state.stats_model.calculate_confidence_intervals()

        if not confidence.get('insufficient_data', False):
            st.write(f"• **Confidence Interval**: {confidence['interpretation']}")

        # Uncertainty summary
        uncertainty_summary = st.session_state.stats_model.get_uncertainty_summary()

        st.write(f"• **Uncertainty Level**: {uncertainty_summary['uncertainty_level']}")
        st.write(f"• {uncertainty_summary['message']}")

        # Show happiness over time
        if len(happiness_values) >= 5:
            fig_time = go.Figure()

            fig_time.add_trace(go.Scatter(
                x=list(range(len(happiness_values))),
                y=happiness_values,
                mode='lines+markers',
                name='Happiness Over Time'
            ))

            # Add confidence band
            mean = moments['mean']
            std = moments['std_deviation']

            x_range = list(range(len(happiness_values)))
            upper_bound = [mean + std] * len(happiness_values)
            lower_bound = [mean - std] * len(happiness_values)

            fig_time.add_trace(go.Scatter(
                x=x_range + x_range[::-1],
                y=upper_bound + lower_bound[::-1],
                fill='toself',
                fillcolor='rgba(255,0,0,0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='±1 Standard Deviation'
            ))

            fig_time.update_layout(
                title="Happiness Trend with Uncertainty Band",
                xaxis_title="Interaction Number",
                yaxis_title="Happiness",
                height=250
            )

            st.plotly_chart(fig_time, use_container_width=True)

    # Educational insights
    st.subheader("📚 Statistical Concepts Explained")
    insights = st.session_state.stats_model.get_educational_insights()

    for concept in insights['statistical_concepts'][:2]:
        st.write(f"• {concept}")


if __name__ == "__main__":
    main()