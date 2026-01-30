from src.IQL.learning.class_state import ClassState
from qiskit.visualization.bloch import Bloch
import numpy as np
import matplotlib.pyplot as plt

class MemoryBank:
    def __init__(self, class_states):
        self.class_states = class_states

    def scores(self, psi):
        return [
            cs.score(psi)
            for cs in self.class_states
        ]

    def increment_age(self):
        """
        Increment age of all memories by 1.
        Call once per training step.
        """
        for cs in self.class_states:
            cs.age += 1

    def update_harm_ema(self, psi,y_true, tau_responsible, beta):
        """
        Update harm EMA for responsible memories.

        Args:
            psi: input state
            tau_responsible: responsibility threshold
            beta: EMA decay factor
        """
        scores = self.scores(psi)

        for cs, s in zip(self.class_states, scores):
            if abs(s) > tau_responsible and cs.label is not None:
                harm = -y_true * s
                cs.harm_ema = beta * cs.harm_ema + (1 - beta) * harm

    def winner(self, psi):
        scores = self.scores(psi)
        idx = int(max(range(len(scores)), key=lambda i: abs(scores[i])))
        #idx = int(max(range(len(scores)), key=lambda i: scores[i])) ## causes lower score ??
        return idx, scores[idx]

    def add_memory(self, chi_vector, backend, label: int):
        """
        Add a new memory to the bank.
        
        Args:
            chi_vector: quantum state vector
            backend: interference backend
            label: class label (mandatory)
        """
        self.class_states.append(ClassState(chi_vector, backend=backend, label=label))

    def remove(self, idx):
        """Remove memory at index idx."""
        if 0 <= idx < len(self.class_states):
            del self.class_states[idx]
    
    def prune(self, prune_states):
        """
        Remove given ClassState objects from the memory bank.
        """
        self.class_states = [
            cs for cs in self.class_states
            if cs not in prune_states
        ]

    def visualize(
        self,
        qubit: int = 0,
        title: str | None = None,
        save_path: str | None = None,
        show: bool = True,
    ):
        """
        Visualize the MEMORY-BANK-LEVEL geometry on a single Bloch sphere.

        - Points  : individual memory states (projected)
        - Arrows  : class centroids
        - Colors  : red = class -1 / 0, blue = class +1 / 1
        - STATIC snapshot (no learning, no dynamics)
        """

        if not self.class_states:
            raise RuntimeError("MemoryBank is empty")

        bloch = Bloch()
        bloch.vector_color = [] 

        red_pts, blue_pts = [], []

        # --- Pauli matrices ---
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        I = np.eye(2, dtype=complex)

        def pauli_on_qubit(P, q, n):
            ops = [I] * n
            ops[q] = P
            out = ops[0]
            for op in ops[1:]:
                out = np.kron(out, op)
            return out

        # --- Project each memory ---
        for cs in self.class_states:
            chi = cs.vector
            n = int(np.log2(len(chi)))
            if 2**n != len(chi):
                raise ValueError("State dimension must be 2^n")

            Xq = pauli_on_qubit(X, qubit, n)
            Yq = pauli_on_qubit(Y, qubit, n)
            Zq = pauli_on_qubit(Z, qubit, n)

            v = np.array([
                float(np.real(np.vdot(chi, Xq @ chi))),
                float(np.real(np.vdot(chi, Yq @ chi))),
                float(np.real(np.vdot(chi, Zq @ chi))),
            ])

            bloch.add_vectors(v)
            bloch.vector_color.append(
                "red" if cs.label in [-1, 0] else "blue"
            )

            if cs.label in [-1, 0]:
                red_pts.append(v)
            else:
                blue_pts.append(v)

        # --- Add centroid arrows ---
        def add_centroid(vectors, color):
            """
            Add a class centroid as an ARROW on the Bloch sphere.

            - vectors : list of Bloch vectors (Nx3)
            - color   : color for the centroid arrow
            """

            if len(vectors) == 0:
                return

            mu = np.mean(vectors, axis=0)
            norm = np.linalg.norm(mu)

            if norm < 1e-9:
                return

            # Keep centroid inside Bloch ball
            mu = mu / max(1.0, norm)

            # --- Temporarily force arrow rendering ---
            previous_style = bloch.vector_style
            bloch.vector_style = "arrow"

            bloch.add_vectors(mu)
            bloch.vector_color.append(color)

            # --- Restore previous style (points) ---
            bloch.vector_style = previous_style


        add_centroid(red_pts, "darkred")
        add_centroid(blue_pts, "darkblue")

        # --- Title / save / show ---
        if title:
            bloch.title = title

        if save_path:
            bloch.save(save_path)

        if show:
            plt.show()
