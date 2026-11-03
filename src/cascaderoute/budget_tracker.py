class BudgetTracker:
    """Tracks token spending and provides budget state."""
    
    def __init__(self, total_budget: int):
        self.total_budget = total_budget
        self.tokens_spent = 0
        
    def add_expense(self, tokens: int):
        """Record spent tokens."""
        self.tokens_spent += tokens
        
    def get_remaining_pct(self) -> float:
        """Returns budget remaining as a percentage [0.0, 1.0]."""
        remaining = self.total_budget - self.tokens_spent
        pct = remaining / self.total_budget
        return max(0.0, pct)
