class Regime4BPruning:
    """
    Regime-4B: Responsible EMA-Based Memory Pruning

    Removes memories that:
    - are old enough
    - are responsible for interference
    - consistently interfere destructively with their own class
    """

    def __init__(
        self,
        memory_bank,
        tau_harm=-0.2,
        min_age=200,
        min_per_class=1,
        prune_interval=200,
    ):
        self.memory_bank = memory_bank
        self.tau_harm = tau_harm
        self.min_age = min_age
        self.min_per_class = min_per_class
        self.prune_interval = prune_interval

        self.step_count = 0
        self.num_pruned = 0

    # -------------------------------------------------
    # Called once per training step
    # -------------------------------------------------
    def step(self):
        self.step_count += 1

        if self.step_count % self.prune_interval != 0:
            return []

        return self.prune()

    # -------------------------------------------------
    # Core pruning logic
    # -------------------------------------------------
    def prune(self):
        to_prune = []

        # Count memories per class
        class_counts = {}
        for cs in self.memory_bank.class_states:
            class_counts.setdefault(cs.label, 0)
            class_counts[cs.label] += 1

        # Identify prune candidates
        for cs in self.memory_bank.class_states:
            if cs.age < self.min_age:
                continue

            if cs.harm_ema < self.tau_harm:
                # enforce class floor
                if class_counts.get(cs.label, 0) > self.min_per_class:
                    to_prune.append(cs)
                    class_counts[cs.label] -= 1

        if to_prune:
            self.memory_bank.prune(to_prune)
            self.num_pruned += len(to_prune)

        return to_prune

    # -------------------------------------------------
    # Diagnostics
    # -------------------------------------------------
    def summary(self):
        return {
            "num_pruned": self.num_pruned,
            "current_memory_size": len(self.memory_bank.class_states),
            "steps": self.step_count,
        }
