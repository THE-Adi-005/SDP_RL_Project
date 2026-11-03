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

