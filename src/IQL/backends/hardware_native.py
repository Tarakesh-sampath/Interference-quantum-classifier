import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit_aer import AerSimulator
import time

class HardwareNativeBackend:
    """
    Hardware-native Hadamard test implementation.
    Computes Re⟨chi | psi⟩ using controlled state-preparation circuits.
    """

    def __init__(self, backend=None, shots=25, seed_simulator=None, basis_gates=None):
        self.backend = backend or AerSimulator()
        self.shots = shots
        self.seed_simulator = seed_simulator
        self.basis_gates = basis_gates

    def _build_circuit(self, chi, psi):
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
        return qc

    def _transpile(self, qc):
        # Transpile against the noisy basis when one is supplied, otherwise the
        # StatePreparation blocks never decompose into gates the noise model
        # attaches to (silent no-op noise).
        if self.basis_gates is not None:
            return transpile(qc, self.backend, basis_gates=self.basis_gates)
        return transpile(qc, self.backend)

    def _counts_to_score(self, counts) -> float:
        n0 = counts.get('0', 0)
        n1 = counts.get('1', 0)
        return float((n0 - n1) / self.shots)

    def score(self, chi, psi) -> float:
        qc = self._transpile(self._build_circuit(chi, psi))
        job = self.backend.run(qc, shots=self.shots, seed_simulator=self.seed_simulator)
        counts = job.result().get_counts()
        return self._counts_to_score(counts)

    def score_batch(self, chi, psis):
        """Estimate Re<chi|psi_j> for many psi against a fixed chi in one run().

        Per-call transpile is the bottleneck for large sweeps; batching submits
        all circuits to a single backend.run().
        """
        circuits = [self._transpile(self._build_circuit(chi, psi)) for psi in psis]
        job = self.backend.run(circuits, shots=self.shots, seed_simulator=self.seed_simulator)
        result = job.result()
        return np.array(
            [self._counts_to_score(result.get_counts(i)) for i in range(len(circuits))],
            dtype=float,
        )