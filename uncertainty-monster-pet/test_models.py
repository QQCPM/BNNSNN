"""
Quick test script to verify all models work correctly
"""

def test_models():
    print("🧪 Testing Uncertainty Monster Pet Models...")

    try:
        # Test Bayesian model
        print("\n🧠 Testing Bayesian Neural Network...")
        from models.bayesian_mood import create_bayesian_mood_predictor
        bayesian_model = create_bayesian_mood_predictor()
        print("✅ Bayesian model created successfully!")

        # Test prediction
        result = bayesian_model.predict_with_uncertainty(0.5, 5, 0.7)
        print(f"   Prediction: {result['mood_prediction']:.3f} ± {result['uncertainty_std']:.3f}")

    except Exception as e:
        print(f"❌ Bayesian model error: {e}")
        return False

    try:
        # Test Temporal SNN
        print("\n⚡ Testing Spiking Neural Network...")
        from models.temporal_energy import create_temporal_energy_snn
        snn_model = create_temporal_energy_snn()
        print("✅ SNN model created successfully!")

        # Test simulation
        from datetime import datetime
        test_interactions = [{'type': 'play', 'timestamp': datetime.now()}]
        result = snn_model.simulate_energy_response(test_interactions)
        print(f"   Energy prediction: {result['energy_prediction']:.3f}")

    except Exception as e:
        print(f"❌ SNN model error: {e}")
        return False

    try:
        # Test Statistical model
        print("\n📊 Testing Statistical Analysis...")
        from models.statistical_happiness import create_happiness_statistics
        stats_model = create_happiness_statistics()
        print("✅ Statistical model created successfully!")

        # Test analysis
        moments = stats_model.compute_basic_moments()
        if not moments.get('insufficient_data', False):
            print(f"   Mean happiness: {moments['mean']:.3f}")
        else:
            print("   Model initialized (needs more data for analysis)")

    except Exception as e:
        print(f"❌ Statistical model error: {e}")
        return False

    print("\n🎉 All models working correctly!")
    print("🚀 Ready to run: python run_demo.py")
    return True

if __name__ == "__main__":
    test_models()