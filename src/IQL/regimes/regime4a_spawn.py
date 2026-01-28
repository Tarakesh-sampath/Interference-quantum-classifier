import numpy as np
from collections import defaultdict

from src.IQL.learning.update import update
from src.IQL.backends.exact import ExactBackend


class Regime4ASpawn:
    """
    Regime-4A: Interference-Coverage Adaptive Memory

    Properties:
    - New implementation (does NOT inherit from Regime-3C)
    - Uses EXACT Regime-3A semantics:
        * Winner-Take-All selection
        * Same misclassification rule
        * Same Regime-2 update
    - Adds memory ONLY when:
        * Winner interference is weak (poor coverage)
        * Winner misclassifies the sample
        * Spawn cooldown allows it
    - Early phase: class-polarized spawning
    - Later phase: class-agnostic spawning
    """

    def __init__(
        self,
        memory_bank,
        eta: float = 0.1,
        backend=None,
        delta_cover: float = 0.2,
        spawn_cooldown: int = 100,
        min_polarized_per_class: int = 1,
    ):
        """
        Args:
            memory_bank (MemoryBank): existing memory bank
            eta (float): learning rate (Regime-2 update)
            backend (InterferenceBackend): scoring backend
            delta_cover (float): minimum |interference| required to avoid spawning
            spawn_cooldown (int): minimum steps between spawns
            min_polarized_per_class (int): bootstrap polarized memories per class
        """
        self.memory_bank = memory_bank
        self.eta = eta
        self.backend = backend or ExactBackend()

        # Regime-4A parameters
        self.delta_cover = delta_cover
        self.spawn_cooldown = spawn_cooldown
        self.min_polarized_per_class = min_polarized_per_class

        # Internal state
        self.steps_since_spawn = spawn_cooldown
        self.polarized_count = defaultdict(int)

        # Logging / diagnostics
        self.num_spawns = 0
        self.num_updates = 0
        self.history = {
            "action": [],        # "spawned" | "updated" | "noop"
            "winner_score": [],
            "memory_size": [],
        }

    # -------------------------------------------------
    # Core step
    # -------------------------------------------------
    def step(self, psi: np.ndarray, y: int):
        """
        Process a single training example.

        Args:
            psi (np.ndarray): input quantum state |psi>
            y (int): label in {-1, +1}

        Returns:
            action (str): "spawned", "updated", or "noop"
        """

        # ---------------------------------------------
        # 1. Compute interference scores
        # ---------------------------------------------
        scores = self.memory_bank.scores(psi)

        if len(scores) == 0:
            raise RuntimeError("MemoryBank is empty — cannot run Regime-4A.")

        # ---------------------------------------------
        # 2. Winner-Take-All (EXACT Regime-3A semantics)
        # ---------------------------------------------
        winner_idx = max(range(len(scores)), key=lambda i: abs(scores[i]))
        s_star = scores[winner_idx]

        # ---------------------------------------------
        # 3. Coverage + misclassification checks
        # ---------------------------------------------
        poor_coverage = abs(s_star) < self.delta_cover
        misclassified = (y * s_star) < 0
        spawn_allowed = self.steps_since_spawn >= self.spawn_cooldown

        # ---------------------------------------------
        # 4. Regime-4A: Spawn new memory if needed
        # ---------------------------------------------
        if poor_coverage and misclassified and spawn_allowed:
            residual = psi.astype(np.complex128, copy=True)

            # Orthogonalize against existing memory
            for cs in self.memory_bank.class_states:
                proj = np.vdot(cs.vector, psi)
                residual -= proj * cs.vector

            norm = np.linalg.norm(residual)

            if norm > 1e-8:
                residual /= norm

                # -------------------------------------
                # Polarized → agnostic transition
                # -------------------------------------
                # Polarized → agnostic transition
                if self.polarized_count[y] < self.min_polarized_per_class:
                    chi_new = y * residual
                    self.polarized_count[y] += 1
                    label = y  # ✅ SET LABEL for polarized memories
                else:
                    chi_new = residual
                    label = None  # Agnostic memories have no label

                self.memory_bank.add_memory(chi_new, self.backend, label=label)  # ✅ PASS LABEL
                self.steps_since_spawn = 0
                self.num_spawns += 1

                self.history["action"].append("spawned")
                self.history["winner_score"].append(float(s_star))
                self.history["memory_size"].append(
                    len(self.memory_bank.class_states)
                )

                return "spawned"

        # ---------------------------------------------
        # 5. Otherwise: standard Regime-3A update
        # ---------------------------------------------
        cs = self.memory_bank.class_states[winner_idx]

        chi_new, updated = update(
            cs.vector, psi, y, self.eta, self.backend
        )

        if updated:
            cs.vector = chi_new
            self.num_updates += 1

        self.steps_since_spawn += 1

        self.history["action"].append("updated" if updated else "noop")
        self.history["winner_score"].append(float(s_star))
        self.history["memory_size"].append(
            len(self.memory_bank.class_states)
        )

        return "updated" if updated else "noop"

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------
    def fit(self, X, y):
        """
        Online training over dataset.

        Args:
            X (Iterable[np.ndarray]): input states
            y (Iterable[int]): labels in {-1, +1}
        """
        for psi, label in zip(X, y):
            self.step(psi, label)
        return self

    # -------------------------------------------------
    # Convenience helpers
    # -------------------------------------------------
    def memory_size(self) -> int:
        return len(self.memory_bank.class_states)

    def summary(self) -> dict:
        return {
            "memory_size": self.memory_size(),
            "num_spawns": self.num_spawns,
            "num_updates": self.num_updates,
        }
