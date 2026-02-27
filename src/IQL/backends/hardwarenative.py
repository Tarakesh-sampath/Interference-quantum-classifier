import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit_aer import AerSimulator


class HardwareNativeBackend:
    """
    Hardware-native Hadamard test implementation.
    Computes Re⟨chi | psi⟩ using controlled state-preparation circuits.
    """

    def __init__(self, backend=None, shots=25):
        self.backend = backend or AerSimulator()
        self.shots = shots

    def score(self, chi, psi) -> float:
        chi = np.asarray(chi, dtype=np.complex128)
        psi = np.asarray(psi, dtype=np.complex128)

        chi = chi / np.linalg.norm(chi)
        psi = psi / np.linalg.norm(psi)

        assert chi.shape == psi.shape
        n = int(np.log2(len(psi)))
        assert 2**n == len(psi)

        qc = QuantumCircuit(1 + n, 1)
        anc = 0
        data = list(range(1, 1 + n))

        psi_state = StatePreparation(psi)
        chi_state = StatePreparation(chi)

        # Prepare |psi⟩
        qc.append(psi_state, data)

        # Hadamard on ancilla
        qc.h(anc)

        # Controlled Uψ†
        qc.append(psi_state.inverse().control(1), [anc] + data)

        # Controlled Uχ
        qc.append(chi_state.control(1), [anc] + data)

        # Final Hadamard
        qc.h(anc)

        # Measure ancilla
        qc.measure(anc, 0)

        # Transpile for backend
        tqc = transpile(qc, self.backend)

        # Execute
        job = self.backend.run(tqc, shots=self.shots)
        counts = job.result().get_counts()

        # Compute expectation value
        n0 = counts.get('0', 0)
        n1 = counts.get('1', 0)

        z_exp = (n0 - n1) / self.shots

        return float(z_exp)