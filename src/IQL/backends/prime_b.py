import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli
from qiskit.circuit.library import StatePreparation

from .base import InterferenceBackend


class PrimeBBackend(InterferenceBackend):
    """
    PrimeB (ISDO-B′) Backend
    -----------------------

    Observable-engineered, decision-sufficient implementation of ISDO.

    Computes:
        S(ψ; χ) = ⟨ψ | U_χ† Z^{⊗n} U_χ | ψ⟩

    Properties:
    - No ancilla qubit
    - No controlled unitaries
    - χ appears only as a basis rotation
    - Fixed, hardware-native observable
    - Preserves sign + ordering (not exact inner product)

    Intended role:
    - Fast inference
    - NISQ-friendly deployment backend
    \"""

    @staticmethod
    def _statevector_to_unitary(state: np.ndarray) -> np.ndarray:
        """
        Construct a unitary U such that:
            U |0...0⟩ = |state⟩

        Uses Gram–Schmidt completion.
        """
        state = np.asarray(state, dtype=np.complex128)
        state = state / np.linalg.norm(state)

        dim = len(state)
        U = np.zeros((dim, dim), dtype=np.complex128)
        U[:, 0] = state

        for i in range(1, dim):
            v = np.zeros(dim, dtype=np.complex128)
            v[i] = 1.0

            for j in range(i):
                v -= np.vdot(U[:, j], v) * U[:, j]

            norm = np.linalg.norm(v)
            if norm < 1e-12:
                v = np.random.randn(dim) + 1j * np.random.randn(dim)
                for j in range(i):
                    v -= np.vdot(U[:, j], v) * U[:, j]
                v /= np.linalg.norm(v)
            else:
                v /= norm

            U[:, i] = v

        return U

    def score(self, chi: np.ndarray, psi: np.ndarray) -> float:
        \"""
        Compute PrimeB interference score.

        Args:
            chi : np.ndarray
                Class memory state |χ⟩
            psi : np.ndarray
                Input state |ψ⟩

        Returns:
            float
                Decision-sufficient interference score
        \"""
        chi = np.asarray(chi, dtype=np.complex128)
        psi = np.asarray(psi, dtype=np.complex128)

        chi /= np.linalg.norm(chi)
        psi /= np.linalg.norm(psi)

        dim = len(psi)
        n = int(np.log2(dim))
        if 2 ** n != dim:
            raise ValueError("State dimension must be a power of 2")

        # Build circuit
        qc = QuantumCircuit(n)

        # Prepare |ψ⟩
        qc.append(StatePreparation(psi), range(n))

        # Apply U_χ
        U_chi = self._statevector_to_unitary(chi)
        qc.unitary(U_chi, range(n), label="U_chi")

        # Evaluate ⟨Z^{⊗n}⟩
        sv = Statevector.from_instruction(qc)
        observable = Pauli("Z"+"I" * (n-1))

        return float(sv.expectation_value(observable).real)
    """
    pass