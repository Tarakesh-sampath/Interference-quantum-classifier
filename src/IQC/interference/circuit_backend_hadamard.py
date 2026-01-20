import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from .base import InterferenceBackend


class HadamardInterferenceBackend(InterferenceBackend):
    """
    Hadamard-test-based interference backend.
    Estimates Re⟨chi | psi⟩ via an ancilla-X expectation.
    Reference (canonical) quantum embodiment.
    """

    def __init__(self, simulator=None):
        # Default to statevector simulator (noise-free reference)
        self.simulator = simulator or AerSimulator(method="statevector")

    @staticmethod
    def _prepare_state(qc, statevector, qubits):
        """
        Prepare |statevector> on 'qubits' using initialize.
        Assumes statevector is normalized.
        """
        qc.initialize(statevector, qubits)

    def score(self, chi, psi) -> float:
        """
        Return Re⟨chi | psi⟩ using a Hadamard test.

        Implementation notes:
        - ancilla qubit index = 0
        - data qubits = 1..n
        - Uses statevector simulation (exact expectation)
        """

        # Sanity checks
        chi = np.asarray(chi, dtype=np.complex128)
        psi = np.asarray(psi, dtype=np.complex128)
        assert chi.shape == psi.shape
        n = int(np.log2(len(psi)))
        assert 2**n == len(psi)

        # Qubits: [ancilla | data...]
        qc = QuantumCircuit(1 + n)

        anc = 0
        data = list(range(1, 1 + n))

        # Prepare |psi> on data
        self._prepare_state(qc, psi, data)

        # Hadamard on ancilla
        qc.h(anc)

        # Controlled-U where U maps |psi> -> |chi>
        # We implement U = |chi><psi| + (I - |psi><psi|)
        # For statevector simulation, this is exact via unitary construction.

        # Build U as a matrix on data qubits
        proj_psi = np.outer(psi, np.conjugate(psi))
        U = np.outer(chi, np.conjugate(psi)) + (np.eye(len(psi)) - proj_psi)

        # Apply controlled-U
        qc.unitary(U, data, label="U_chi_psi").control(1)

        # Hadamard on ancilla
        qc.h(anc)

        # Get final statevector
        sv = Statevector.from_instruction(qc)

        # Compute ⟨X_anc⟩ = Re⟨chi|psi⟩
        # X expectation = sum_{z} (-1)^{z_anc} |amp_z|^2
        exp_x = 0.0
        for i, amp in enumerate(sv.data):
            bit = (i >> n) & 1  # ancilla is MSB
            exp_x += (1 if bit == 0 else -1) * (abs(amp) ** 2)

        return float(exp_x)
