import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli
from qiskit.circuit.library import UnitaryGate

from .base import InterferenceBackend


class ISDOBPrimeInterferenceBackend(InterferenceBackend):
    """
    ISDO-B′ interference backend.

    Implements the observable:
        S_ISDO(psi; chi) = <psi | U_chi^† Z^{⊗n} U_chi | psi>

    Properties:
    - No controlled unitaries
    - Fixed measurement (Z^{⊗n})
    - chi appears only as a basis change (U_chi)
    - Hardware-friendly compared to Hadamard test
    """

    def __init__(self):
        # Optional cache for U_chi to avoid recomputation
        self._cache = {}

    @staticmethod
    def _statevector_to_unitary(vec):
        """
        Build a unitary U such that U |0...0> = |vec>.
        Completed via Gram–Schmidt to a full unitary.
        """
        vec = np.asarray(vec, dtype=np.complex128)
        vec = vec / np.linalg.norm(vec)
        dim = len(vec)

        U = np.zeros((dim, dim), dtype=np.complex128)
        U[:, 0] = vec

        for i in range(1, dim):
            v = np.zeros(dim, dtype=np.complex128)
            v[i] = 1.0

            for j in range(i):
                v -= np.vdot(U[:, j], v) * U[:, j]

            nrm = np.linalg.norm(v)
            if nrm < 1e-12:
                v = np.random.randn(dim) + 1j * np.random.randn(dim)
                for j in range(i):
                    v -= np.vdot(U[:, j], v) * U[:, j]
                v /= np.linalg.norm(v)
            else:
                v /= nrm

            U[:, i] = v

        return U

    def _get_U_chi(self, chi):
        """
        Cached construction of U_chi.
        """
        key = chi.tobytes()
        if key not in self._cache:
            U = self._statevector_to_unitary(chi)
            self._cache[key] = UnitaryGate(U)
        return self._cache[key]

    def score(self, chi, psi) -> float:
        chi = np.asarray(chi, dtype=np.complex128)
        psi = np.asarray(psi, dtype=np.complex128)

        chi = chi / np.linalg.norm(chi)
        psi = psi / np.linalg.norm(psi)

        assert chi.shape == psi.shape
        n = int(np.log2(len(psi)))
        assert 2**n == len(psi)

        qc = QuantumCircuit(n)

        # Prepare |psi>
        qc.initialize(psi, list(range(n)))

        # Apply U_chi
        U_chi = self._get_U_chi(chi)
        qc.append(U_chi, list(range(n)))

        # Measure Z^{⊗n} expectation exactly (statevector)
        sv = Statevector.from_instruction(qc)
        Z_all = Pauli("Z" + "I" * (n - 1))

        exp_val = sv.expectation_value(Z_all).real
        return float(exp_val)
