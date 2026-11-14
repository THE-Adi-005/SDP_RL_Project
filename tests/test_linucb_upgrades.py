"""Unit tests for LinUCB warm-start and budget-annealed exploration."""
import numpy as np
import pytest
from src.rl.linucb import LinUCB


def test_linucb_warm_start():
    np.random.seed(42)
    n_actions = 4
    d = 16
    N = 100

    states = np.random.randn(N, d).astype(np.float32)
    actions = np.random.randint(0, n_actions, size=N)
    # Ground truth: action 2 gives high reward for positive states
    rewards = (actions == 2).astype(np.float32) * 1.0 + np.random.randn(N) * 0.05

    agent = LinUCB(n_actions=n_actions, alpha=0.1)
    agent.warm_start(states, actions, rewards)

    assert agent.warm_started
    assert agent._initialized
    assert agent.feature_dim == d
    for a in range(n_actions):
        assert agent.A[a].shape == (d, d)
        assert agent.b[a].shape == (d,)
        assert agent.A_inv[a].shape == (d, d)

    # After warm starting on action 2 having high reward, test state should favor action 2
    test_state = np.ones(d, dtype=np.float32)
    selected = agent.select_action(test_state)
    assert selected == 2, f"Expected warm-started agent to pick arm 2, got {selected}"


def test_linucb_alpha_anneal():
    n_actions = 3
    d = 8
    agent = LinUCB(n_actions=n_actions, alpha=2.0, alpha_anneal=True)
    state = np.ones(d, dtype=np.float32)

    # When budget fraction is 1.0, exploration is full (alpha = 2.0)
    # When budget fraction is 0.0, exploration is zero (pure exploitation)
    act1 = agent.select_action(state, budget_fraction=1.0)
    act2 = agent.select_action(state, budget_fraction=0.0)
    assert 0 <= act1 < n_actions
    assert 0 <= act2 < n_actions


def test_linucb_score_arms_and_matrix_warmstart():
    np.random.seed(7)
    N, d, K = 60, 12, 4
    X = np.random.randn(N, d).astype(np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    R = np.random.rand(N, K).astype(np.float32) * 0.4
    R[:, 1] += 0.5  # Arm 1 dominates

    agent = LinUCB(n_actions=K, alpha=0.2, feature_dim=d, alpha_anneal=True, use_gpu=False)
    agent.warm_start_from_full_matrix(X, R, regularization=2.0, sample_weight=0.5)
    preds, uncs, ucb = agent.score_arms(X[0], budget_fraction=0.8)
    assert preds.shape == (K,) and uncs.shape == (K,) and ucb.shape == (K,)
    assert np.all(uncs >= 0.0)
    assert int(np.argmax(preds)) == 1


def test_linucb_alpha_min_and_budget_guardrail():
    d, K = 8, 3
    agent = LinUCB(n_actions=K, alpha=1.0, alpha_min=0.15, alpha_anneal=True, feature_dim=d, use_gpu=False)
    x = np.ones(d, dtype=np.float32) / np.sqrt(d)
    preds, uncs, ucb_zero = agent.score_arms(x, budget_fraction=0.0)
    assert np.allclose(ucb_zero, preds + 0.15 * uncs, atol=1e-6)

    arm_costs = np.array([0.01, 0.25, 0.50])
    # Remaining budget < reserve_multiplier * c_min forces cheapest arm 0 deterministically
    act, forced = agent.select_action_budgeted(x, budget_remaining_usd=0.005, total_budget_usd=1.0, arm_cost_est=arm_costs)
    assert forced is True and act == 0

    # Remaining budget 0.10 masks out 0.25 and 0.50 -> picks feasible arm 0
    act2, forced2 = agent.select_action_budgeted(x, budget_remaining_usd=0.10, total_budget_usd=1.0, arm_cost_est=arm_costs)
    assert act2 == 0 and forced2 is False


def test_linucb_discounted_sliding_window_and_onboarding():
    np.random.seed(99)
    d, K = 8, 3
    x = np.ones(d, dtype=np.float32) / np.sqrt(d)
    sw = LinUCB(n_actions=K, alpha=0.1, feature_dim=d, sliding_window=10, use_gpu=False)
    disc = LinUCB(n_actions=K, alpha=0.1, feature_dim=d, discount_factor=0.95, use_gpu=False)
    for i in range(15):
        sw.update(x, 0, 0.9 if i >= 5 else 0.1)
        disc.update(x, 0, 0.9 if i >= 5 else 0.1)
    assert len(sw._sw_buffers[0]) == 10
    p_sw, _, _ = sw.score_arms(x)
    p_d, _, _ = disc.score_arms(x)
    assert p_sw[0] > 0.5 and p_d[0] > 0.5

    # Hot onboarding resets arm 2 cleanly with high uncertainty and learns online
    disc.hot_onboard_arm(arm=2, prior_strength=0.0)
    _, uncs_new, _ = disc.score_arms(x)
    assert uncs_new[2] > uncs_new[0]


