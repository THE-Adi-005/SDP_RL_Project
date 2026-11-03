"""Unit tests for newly implemented baselines:
- MHRRouter (retrieval-enhanced)
- ARMSRouter (profile-augmented)
- OfflineCostAwareRouter (soft-target offline)
- FrugalGPTCascade (sequential cascade)
- ConfidenceGate (visual perturbation & overthinking scoring)
"""
import numpy as np
import pytest
from src.rl.mhr_router import MHRRouter
from src.rl.arms_router import ARMSRouter
from src.rl.offline_router import OfflineCostAwareRouter
from src.rl.frugalgpt_cascade import FrugalGPTCascade
from src.cascaderoute.confidence_gate import ConfidenceGate


def test_mhr_router_basic():
    router = MHRRouter(n_actions=4, k=3)
    state = np.ones(10, dtype=np.float32)

    # Initial selection before any memory
    act = router.select_action(state)
    assert 0 <= act < 4

    # Update with success on arm 2
    for _ in range(5):
        router.update(state, 2, 1.0)
    # Update with failure on arm 0
    for _ in range(5):
        router.update(state, 0, 0.0)

    # Now should strongly prefer arm 2
    assert router.select_action(state) == 2


def test_arms_router_fit_and_predict():
    router = ARMSRouter(n_actions=4)
    N, d = 50, 16
    X = np.random.randn(N, d).astype(np.float32)
    # Artificial labels
    y = np.random.randint(0, 4, size=N)

    router.fit(X, y)
    assert router.is_fitted

    pred = router.select_action(X[0])
    assert 0 <= pred < 4
    probs = router.predict_proba(X[0])
    assert len(probs) == 4
    assert np.isclose(np.sum(probs), 1.0)


def test_offline_cost_aware_router():
    router = OfflineCostAwareRouter(n_actions=4, lambda_weight=0.5)
    N, d = 60, 12
    X = np.random.randn(N, d)
    Y = np.zeros((N, 4))
    C = np.ones((N, 4)) * 10.0

    # Arm 1 is correct for first half, arm 2 is correct for second half
    Y[:30, 1] = 1.0
    Y[30:, 2] = 1.0

    router.fit_from_matrices(X, Y, C)
    assert router.is_fitted
    pred = router.select_action(X[0])
    assert pred in [1, 2]


def test_frugalgpt_cascade():
    cascade = FrugalGPTCascade(n_actions=4, acceptance_threshold=0.8)
    # Query where arm 0 is correct with high confidence
    outcomes_easy = {
        0: {"correct": True, "total_tokens": 50.0, "latency": 0.2, "confidence": 0.95},
        1: {"correct": True, "total_tokens": 150.0, "latency": 0.5, "confidence": 0.98}
    }
    arm, is_corr, tok, lat, esc = cascade.execute_cascade(outcomes_easy)
    assert arm == 0
    assert is_corr is True
    assert esc == 0
    assert tok == 50.0

    # Query where arm 0 fails with low confidence, escalating to arm 1
    outcomes_hard = {
        0: {"correct": False, "total_tokens": 50.0, "latency": 0.2, "confidence": 0.3},
        1: {"correct": True, "total_tokens": 150.0, "latency": 0.5, "confidence": 0.90}
    }
    arm, is_corr, tok, lat, esc = cascade.execute_cascade(outcomes_hard)
    assert arm == 1
    assert is_corr is True
    assert esc == 1
    assert tok == 200.0  # Incurred both arm 0 and arm 1 tokens


def test_confidence_gate_overthinking_score():
    gate = ConfidenceGate(tau=0.25)
    # Identical answers -> zero score
    assert gate.compute_overthinking_score("The answer is 42", "The answer is 42") == 0.0
    # Completely divergent answers -> high score
    assert gate.compute_overthinking_score("The answer is 42", "Error invalid input") > 0.5
