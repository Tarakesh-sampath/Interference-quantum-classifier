"""
Standalone visualization for Static ISDO classifier (split diagnostics).

PLOT A:
- Prototypes (class 0 / class 1)
- χ (interference state)

PLOT B:
- χ (interference state)
- Test input

Both plots:
- One figure
- One Bloch sphere per qubit
- Uses ONLY public members of StaticISDOClassifier
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization.bloch import Bloch

from src.utils.paths import load_paths
from src.IQL.baselines.static_isdo_classifier import StaticISDOClassifier


# =================================================
# Pauli utilities (identical to MemoryBank)
# =================================================
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


def bloch_projection(state, qubit: int):
    """
    Safe projection of a single n-qubit pure state
    onto Bloch coordinates of one qubit.
    """
    # Define Pauli matrices locally
    X_local = np.array([[0, 1], [1, 0]], dtype=complex)
    Y_local = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z_local = np.array([[1, 0], [0, -1]], dtype=complex)
    I_local = np.eye(2, dtype=complex)
    
    def pauli_on_qubit_local(P, q, n):
        ops = [I_local] * n
        ops[q] = P
        out = ops[0]
        for op in ops[1:]:
            out = np.kron(out, op)
        return out
    
    state = np.asarray(state, dtype=np.complex128)

    if state.ndim != 1:
        raise ValueError(
            f"bloch_projection expects a single statevector, got shape {state.shape}"
        )

    state /= np.linalg.norm(state)

    n = int(np.log2(len(state)))
    if 2**n != len(state):
        raise ValueError("State dimension must be 2^n")

    Xq = pauli_on_qubit_local(X_local, qubit, n)
    Yq = pauli_on_qubit_local(Y_local, qubit, n)
    Zq = pauli_on_qubit_local(Z_local, qubit, n)

    return np.array([
        float(np.real(np.vdot(state, Xq @ state))),
        float(np.real(np.vdot(state, Yq @ state))),
        float(np.real(np.vdot(state, Zq @ state))),
    ])

# =================================================
# PLOT A: Prototypes + χ
# =================================================
def plot_prototypes_and_chi(
    model,
    title: str,
    save_path: str | None = None,
    show: bool = True,
):
    """
    Visualize:
    - class 0 prototypes
    - class 1 prototypes
    - χ (interference state)

    NO test input.
    """

    # infer dimension from χ
    chi = np.asarray(model.chi, dtype=np.complex128)
    chi /= np.linalg.norm(chi)
    n_qubits = int(np.log2(len(chi)))

    fig = plt.figure(figsize=(4 * n_qubits, 4))

    for q in range(n_qubits):
        ax = fig.add_subplot(1, n_qubits, q + 1, projection="3d")
        bloch = Bloch(fig=fig, axes=ax)
        bloch.vector_color = []

        # ---- prototypes ----
        for label, proto_list in model.prototypes.items():
            color = "red" if label in [0, -1] else "blue"

            for proto in proto_list:
                proto = np.asarray(proto, dtype=np.complex128)
                if proto.ndim > 1:
                    proto = proto.reshape(-1)

                if proto.shape != chi.shape:
                    continue

                proto /= np.linalg.norm(proto)
                v = bloch_projection(proto, q)
                bloch.add_vectors(v)
                bloch.vector_color.append(color)

        # ---- χ arrow ----
        chi_vec = bloch_projection(chi, q)
        prev = bloch.vector_style
        bloch.vector_style = "arrow"
        bloch.add_vectors(chi_vec)
        bloch.vector_color.append("green")
        bloch.vector_style = prev

        ax.set_title(f"Qubit {q}")
        bloch.render()
    fig.suptitle(title, fontsize=14)

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)


# =================================================
# PLOT B: χ + Test Input
# =================================================
def plot_chi_and_test(
    model,
    test_state,
    title: str,
    save_path: str | None = None,
    show: bool = True,
):
    """
    Visualize:
    - χ (interference state)
    - test input

    NO prototypes.
    """

    chi = np.asarray(model.chi, dtype=np.complex128)
    chi /= np.linalg.norm(chi)

    test_state = np.asarray(test_state, dtype=np.complex128)
    test_state /= np.linalg.norm(test_state)

    n_qubits = int(np.log2(len(chi)))

    fig = plt.figure(figsize=(4 * n_qubits, 4))

    for q in range(n_qubits):
        ax = fig.add_subplot(1, n_qubits, q + 1, projection="3d")
        bloch = Bloch(fig=fig, axes=ax)
        bloch.vector_color = []

        # ---- χ arrow ----
        chi_vec = bloch_projection(chi, q)
        prev = bloch.vector_style
        bloch.vector_style = "arrow"
        bloch.add_vectors(chi_vec)
        bloch.vector_color.append("green")
        bloch.vector_style = prev

        # ---- test point ----
        test_vec = bloch_projection(test_state, q)
        bloch.add_vectors(test_vec)
        bloch.vector_color.append("black")

        ax.set_title(f"Qubit {q}")
        bloch.render()
    fig.suptitle(title, fontsize=14)

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)


# =================================================
# MAIN
# =================================================
if __name__ == "__main__":

    _, PATHS = load_paths()

    EMBED_DIR = PATHS["embeddings"]
    PROTO_DIR = PATHS["class_prototypes"]
    K = int(PATHS["class_count"]["K"])

    X = np.load(os.path.join(EMBED_DIR, "val_embeddings.npy"))
    y = np.load(os.path.join(EMBED_DIR, "val_labels.npy"))
    test_idx = np.load(os.path.join(EMBED_DIR, "split_test_idx.npy"))

    X_test = X[test_idx]
    y_test = y[test_idx]

    test_state_0 = X_test[np.where(y_test == 0)[0][0]]
    test_state_1 = X_test[np.where(y_test == 1)[0][0]]

    # ---- Static ISDO ----
    model = StaticISDOClassifier(PROTO_DIR, K)
    model.predict_one(test_state_0)  # populate χ + prototypes

    print(model.chi.shape)
    print(test_state_0.shape)
    print(model.prototypes[0][0].shape)
    print(model.prototypes[1][0].shape)

    # ---- Plot A ----
    plot_prototypes_and_chi(
        model,
        title="Static ISDO – Prototypes and χ",
        save_path="static_isdo_prototypes_chi.png",
        show=True,
    )

    # ---- Plot B ----
    plot_chi_and_test(
        model,
        test_state_0,
        title="Static ISDO – χ and Test Input",
        save_path="static_isdo_chi_test.png",
        show=True,
    )
    
    plot_chi_and_test(
        model,
        test_state_1,
        title="Static ISDO – χ and Test Input",
        save_path="static_isdo_chi_test.png",
        show=True,
    )
