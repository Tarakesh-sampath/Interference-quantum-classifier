import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import UnitaryGate

from .base import InterferenceBackend


class ISDOBPrimeV2InterferenceBackend(InterferenceBackend):
    """
    ISDO-B′ v2: Contrastive projector observable.

    Implements:
        S = <psi | U_chi^† (|0><0| - Pi_perp) U_chi | psi>
          = 2 * |<chi|psi>|^2 - 1     (with alpha = 1)

    Measurement procedure:
      1) Prepare |psi>
      2) Apply U_chi
      3) Measure probability p0 of |0...0>
      4) Return S = 2*p0 - 1

    No ancilla. No controlled unitaries.
    """

    def __init__(self):
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

        # Rotate basis with U_chi
        U_chi = self._get_U_chi(chi)
        qc.append(U_chi, list(range(n)))

        # Exact statevector
        sv = Statevector.from_instruction(qc)

        # Probability of |0...0>
        p0 = abs(sv.data[0]) ** 2

        # Contrastive projector score (alpha = 1)
        return float(2.0 * p0 - 1.0)
