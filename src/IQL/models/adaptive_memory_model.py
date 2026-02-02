# src/IQL/training/adaptive_memory_trainer.py
from src.IQL.regimes.regime4a_spawn import Regime4ASpawn
from src.IQL.regimes.regime4b_pruning import Regime4BPruning
from src.IQL.regimes.regime3a_wta import WinnerTakeAll
from src.utils.paths import load_paths
import os

class AdaptiveMemoryModel:
    """
    System-level adaptive controller for IQC.

    Responsibilities:
    - Enforce memory lifecycle invariants
    - Orchestrate learning (Regime-4A)
    - Handle aging, harm tracking, and pruning (Regime-4B)

    This class contains NO learning logic.
    """

    def __init__(
        self,
        memory_bank,
        learner : Regime4ASpawn,          # Regime4ASpawn
        pruner : Regime4BPruning,           # Regime4BPruning
        tau_responsible: float = 0.1,
        beta: float = 0.98,
        frames_dir: str = None,
        fps: int = 50,
    ):
        self.memory_bank = memory_bank
        self.learner = learner
        self.pruner = pruner
        self.tau_responsible = tau_responsible
        self.beta = beta

        self.step_count = 0
        self.history = {
            "action": [],        # spawned | updated | noop
            "memory_size": [],
            "num_pruned": [],
        }
        self.frames_dir = frames_dir
        self.FRAME_EVERY = fps
    def consolidate(
        self,
        X,
        y,
        epochs: int = 5,
        eta_scale: float = 0.3,
    ):
        """
        Post-adaptive consolidation phase.

        - Freezes memory structure (no spawn, no prune)
        - Refines existing memories using WTA updates
        - Improves margins and accuracy

        Args:
            X: input states
            y: true labels (polar)
            epochs: number of consolidation passes
            eta_scale: scale factor for learning rate
        """

        #print(
        #    f"\n🔒 Consolidation phase started "
        #    f"(epochs={epochs}, eta_scale={eta_scale})"
        #)

        # Winner-Take-All learner (Regime-3A semantics)
        consolidator = WinnerTakeAll(
            memory_bank=self.memory_bank,
            eta=self.learner.eta * eta_scale,
            backend=self.learner.backend,
            alpha=0.0,   # update only on error
            beta=1.0,    # full update
        )

        # IMPORTANT: freeze adaptive structure
        original_spawn = self.learner.step
        original_prune = self.pruner.step

        try:
            # Disable spawn & prune
            self.learner.step = lambda psi, y: "noop"
            self.pruner.step = lambda: []

            for ep in range(epochs):
                consolidator.fit(X, y)
                #print(f"  ✔ Consolidation epoch {ep+1}/{epochs}")

        finally:
            # Restore adaptive behavior
            self.learner.step = original_spawn
            self.pruner.step = original_prune

        #print("🔓 Consolidation phase completed\n")
        return self


    def step(self, psi, y, frames=False):
        """
        Execute ONE adaptive training step.

        Ordering is STRICT and MUST NOT be changed.
        """

        # -------------------------------------------------
        # STEP 1: learning + possible memory spawn
        # -------------------------------------------------
        action = self.learner.step(psi, y)

        # -------------------------------------------------
        # STEP 2: age update (MANDATORY)
        # -------------------------------------------------
        self.memory_bank.increment_age()

        # -------------------------------------------------
        # STEP 3: harm EMA update (MANDATORY)
        # -------------------------------------------------
        self.memory_bank.update_harm_ema(
            psi,
            y_true=y,
            tau_responsible=self.tau_responsible,
            beta=self.beta,
        )

        # -------------------------------------------------
        # STEP 4: pruning (PERIODIC)
        # -------------------------------------------------
        pruned = self.pruner.step()
        num_pruned = len(pruned) if pruned else 0

        # -------------------------------------------------
        # STEP 5: bookkeeping
        # -------------------------------------------------
        self.step_count += 1

        if frames:
            if self.step_count % self.FRAME_EVERY == 0:
                frame_path = f"{self.frames_dir}/frame_{self.step_count:05d}.png"
                self.memory_bank.visualize(
                    qubit=0,
                    title="Adaptive IQC – Memory States (Final Snapshot)",
                    save_path=frame_path,
                show=False,
            )
            
        self.history["action"].append(action)
        self.history["memory_size"].append(
            len(self.memory_bank.class_states)
        )
        self.history["num_pruned"].append(num_pruned)

        return action, pruned

    def fit(self, X, y):
        """
        Online adaptive training loop.
        """
        if self.frames_dir is None:
            print("No frames directory specified. Skipping frame saving.")
            frames = False
        else:
            frames = True
            os.makedirs(self.frames_dir, exist_ok=True)
        for psi, label in zip(X, y):
            self.step(psi, label,frames)
        return self

    def predict(self, X):
        """
        Inference using winner-take-all interference.
        """
        preds = []
        for psi in X:
            _, score = self.memory_bank.winner(psi)
            preds.append(1 if score >= 0 else -1)
        return preds

    def summary(self):
        return {
            "steps": self.step_count,
            "final_memory_size": len(self.memory_bank.class_states),
            "num_spawns": self.history["action"].count("spawned"),
            "num_pruned": sum(self.history["num_pruned"]),
        }
