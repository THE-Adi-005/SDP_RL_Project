def make_always_policy(action_idx: int):
    def select(state):
        return action_idx
    return select
