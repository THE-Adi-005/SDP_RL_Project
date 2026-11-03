"""Unit tests for all 5 RL algorithms.

Tests verify:
- Correct action shape / range
- Weight updates change internal state
- Exploration behavior
- LinUCB matrix dimensions
- DQN replay buffer
- Thompson Sampling posterior updates
"""
import numpy as np
import pytest


# --- Epsilon-Greedy Bandit ---

def test_epsilon_greedy_action_range():
    from src.rl.bandit import EpsilonGreedyBandit
    agent = EpsilonGreedyBandit(n_actions=4, epsilon=0.5)
    state = np.random.randn(10).astype(np.float32)
    for _ in range(20):
        action = agent.select_action(state)
        assert 0 <= action < 4, f"Action {action} out of range"


def test_epsilon_greedy_update_changes_weights():
    from src.rl.bandit import EpsilonGreedyBandit
    agent = EpsilonGreedyBandit(n_actions=4, epsilon=0.0)  # greedy
    state = np.ones(10, dtype=np.float32)
    agent.update(state, 0, 1.0)
    agent.update(state, 1, -1.0)
    assert agent.weights is not None
    # Action 0 got positive reward, should have higher score
    scores = agent.weights @ state
    assert scores[0] > scores[1], "Positive reward arm should score higher"


def test_epsilon_greedy_has_name():
    from src.rl.bandit import EpsilonGreedyBandit
    agent = EpsilonGreedyBandit()
    assert agent.get_name() == "EpsilonGreedy"


# --- UCB1 ---

def test_ucb1_tries_all_arms_first():
    from src.rl.ucb import UCB1
    agent = UCB1(n_actions=4)
    state = np.zeros(10)
    seen = set()
    for _ in range(4):
        action = agent.select_action(state)
        agent.update(state, action, 1.0)
        seen.add(action)
    assert len(seen) == 4, "UCB1 should try all arms before repeating"


def test_ucb1_favors_high_reward_arm():
    from src.rl.ucb import UCB1
    agent = UCB1(n_actions=4, c=0.01)  # low exploration
    state = np.zeros(10)
    # Initialize all arms
    for a in range(4):
        agent.update(state, a, 0.0)
    # Give arm 2 high reward many times
    for _ in range(50):
        agent.update(state, 2, 1.0)
    action = agent.select_action(state)
    assert action == 2, "UCB1 should favor high-reward arm with low c"


def test_ucb1_has_name():
    from src.rl.ucb import UCB1
    assert UCB1().get_name() == "UCB1"


# --- LinUCB ---

def test_linucb_action_range():
    from src.rl.linucb import LinUCB
    agent = LinUCB(n_actions=4, alpha=1.0)
    state = np.random.randn(389).astype(np.float32)
    for _ in range(10):
        action = agent.select_action(state)
        assert 0 <= action < 4


def test_linucb_matrix_dimensions():
    from src.rl.linucb import LinUCB
    agent = LinUCB(n_actions=4, alpha=1.0)
    state = np.random.randn(20).astype(np.float32)
    agent.select_action(state)  # triggers initialization
    assert agent.A[0].shape == (20, 20)
    assert agent.b[0].shape == (20,)
    assert agent.A_inv[0].shape == (20, 20)


def test_linucb_update_changes_matrices():
    from src.rl.linucb import LinUCB
    agent = LinUCB(n_actions=4, alpha=1.0)
    state = np.random.randn(10).astype(np.float32)
    agent.select_action(state)
    A_before = agent.A[0].copy()
    agent.update(state, 0, 1.0)
    assert not np.allclose(agent.A[0], A_before), "A should change after update"


def test_linucb_has_name():
    from src.rl.linucb import LinUCB
    agent = LinUCB(alpha=2.0)
    assert "LinUCB" in agent.get_name()


# --- Thompson Sampling ---

def test_thompson_action_range():
    from src.rl.thompson import ThompsonSamplingLinear
    agent = ThompsonSamplingLinear(n_actions=4, v_squared=1.0)
    state = np.random.randn(20).astype(np.float32)
    for _ in range(10):
        action = agent.select_action(state)
        assert 0 <= action < 4


def test_thompson_posterior_updates():
    from src.rl.thompson import ThompsonSamplingLinear
    agent = ThompsonSamplingLinear(n_actions=4)
    state = np.random.randn(10).astype(np.float32)
    agent.select_action(state)  # initialize
    mu_before = agent.mu[0].copy()
    agent.update(state, 0, 1.0)
    assert not np.allclose(agent.mu[0], mu_before), "Posterior mean should update"


def test_thompson_has_name():
    from src.rl.thompson import ThompsonSamplingLinear
    agent = ThompsonSamplingLinear(v_squared=0.5)
    assert "ThompsonSampling" in agent.get_name()


# --- DQN ---

def test_dqn_action_range():
    from src.rl.dqn import DQNAgent
    agent = DQNAgent(n_actions=4)
    state = np.random.randn(20).astype(np.float32)
    for _ in range(10):
        action = agent.select_action(state)
        assert 0 <= action < 4


def test_dqn_replay_buffer_stores():
    from src.rl.dqn import ReplayBuffer
    buf = ReplayBuffer(capacity=100)
    for i in range(10):
        buf.push(np.array([i, i+1]), i % 4, float(i))
    assert len(buf) == 10


def test_dqn_replay_buffer_max_capacity():
    from src.rl.dqn import ReplayBuffer
    buf = ReplayBuffer(capacity=5)
    for i in range(10):
        buf.push(np.array([i]), 0, 0.0)
    assert len(buf) == 5, "Buffer should cap at capacity"


def test_dqn_epsilon_decays():
    from src.rl.dqn import DQNAgent
    agent = DQNAgent(n_actions=4, epsilon=0.5, epsilon_decay=0.9,
                      batch_size=2)
    state = np.random.randn(10).astype(np.float32)
    eps_before = agent.epsilon
    # Add enough samples and trigger training
    for _ in range(5):
        agent.update(state, 0, 1.0)
    assert agent.epsilon < eps_before, "Epsilon should decay after updates"


def test_dqn_has_name():
    from src.rl.dqn import DQNAgent
    assert DQNAgent().get_name() == "DQN"


# --- Cross-cutting: all agents have consistent interface ---

def test_all_agents_share_interface():
    """Verify all agents have select_action, update, and get_name."""
    from src.rl.bandit import EpsilonGreedyBandit
    from src.rl.ucb import UCB1
    from src.rl.linucb import LinUCB
    from src.rl.thompson import ThompsonSamplingLinear
    from src.rl.dqn import DQNAgent

    agents = [
        EpsilonGreedyBandit(n_actions=4),
        UCB1(n_actions=4),
        LinUCB(n_actions=4),
        ThompsonSamplingLinear(n_actions=4),
        DQNAgent(n_actions=4),
    ]

    state = np.random.randn(20).astype(np.float32)
    for agent in agents:
        assert hasattr(agent, "select_action"), f"{agent} missing select_action"
        assert hasattr(agent, "update"), f"{agent} missing update"
        assert hasattr(agent, "get_name"), f"{agent} missing get_name"

        action = agent.select_action(state)
        assert isinstance(action, int)
        assert 0 <= action < 4

        agent.update(state, action, 0.5)
        name = agent.get_name()
        assert isinstance(name, str) and len(name) > 0
