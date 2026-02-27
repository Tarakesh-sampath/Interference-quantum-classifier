import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, Pauli
from scipy.optimize import minimize
from qiskit_aer import Aer
from .base import InterferenceBackend


class HardwareNativeBackend(InterferenceBackend):
    """
    Hardware-aware Hadamard transition test.

    Public API now supports BOTH:
        score(psi_vector, chi_vector)
        score(psi_circuit, chi_circuit)

    If numpy arrays are passed → internal encoder is used.
    If circuits are passed → used directly.
    """

    def __init__(self, backend=None, shots=256, use_statevector=False):
        self.backend = backend
        self.shots = shots
        self.use_statevector = use_statevector

    # --------------------------------------------------
    # Internal Encoder (Vector → Circuit)
    # --------------------------------------------------
    def _build_ansatz(self, n, params, depth):
        qc = QuantumCircuit(n)
        idx = 0

        for _ in range(depth):
            # Single qubit rotations
            for q in range(n):
                qc.ry(params[idx], q)
                idx += 1

            # Linear entanglement
            for q in range(n - 1):
                qc.cx(q, q + 1)

        return qc

    def _encoder(self, vec, depth=1, maxiter=100):
        """
        Approximate a target statevector using a shallow
        Ry + CX ansatz.
        """

        vec = np.asarray(vec, dtype=np.complex128)
        vec = vec / np.linalg.norm(vec)

        n = int(np.log2(len(vec)))
        assert 2**n == len(vec)

        num_params = depth * n

        def loss(params):
            qc = self._build_ansatz(n, params, depth)
            sv = Statevector.from_instruction(qc)
            fidelity = np.abs(np.vdot(vec, sv.data))**2
            return 1 - fidelity

        init = np.random.uniform(0, 2*np.pi, num_params)

        result = minimize(loss, init, method="COBYLA",
                        options={"maxiter": maxiter})

        trained_qc = self._build_ansatz(n, result.x, depth)
        return trained_qc

    # --------------------------------------------------
    # Main Score Function
    # --------------------------------------------------

    def score(self, psi, chi) -> float:
        """
        psi, chi can be:
            - numpy.ndarray (statevectors)
            - QuantumCircuit (state preparation circuits)
        """

        # --------------------------------------------
        # Convert inputs if necessary
        # --------------------------------------------

        if isinstance(psi, np.ndarray):
            psi_circuit = self._encoder(psi)
        elif isinstance(psi, QuantumCircuit):
            psi_circuit = psi
        else:
            raise TypeError("psi must be numpy.ndarray or QuantumCircuit")

        if isinstance(chi, np.ndarray):
            chi_circuit = self._encoder(chi)
        elif isinstance(chi, QuantumCircuit):
            chi_circuit = chi
        else:
            raise TypeError("chi must be numpy.ndarray or QuantumCircuit")

        assert psi_circuit.num_qubits == chi_circuit.num_qubits
        n = psi_circuit.num_qubits

        # --------------------------------------------
        # Build interference circuit
        # --------------------------------------------

        qc = QuantumCircuit(1 + n, 1)

        anc = 0
        data = list(range(1, n + 1))

        # 1️⃣ Hadamard on ancilla
        qc.h(anc)

        # 2️⃣ Controlled U_psi
        qc.append(psi_circuit.control(1), [anc] + data)

        # 3️⃣ Controlled V_chi^†
        qc.append(chi_circuit.inverse().control(1), [anc] + data)

        # 4️⃣ Final Hadamard
        qc.h(anc)

        # --------------------------------------------
        # Execution Modes
        # --------------------------------------------

        if self.use_statevector:
            sv = Statevector.from_instruction(qc)
            z_exp = sv.expectation_value(Pauli('Z'), [anc]).real
            return float(z_exp)

        qc.measure(anc, 0)

        backend = self.backend or Aer.get_backend("aer_simulator_statevector_gpu")
        tqc = transpile(qc, backend)
        job = backend.run(tqc, shots=self.shots)
        counts = job.result().get_counts()

        p0 = counts.get('0', 0) / self.shots
        p1 = counts.get('1', 0) / self.shots

        return float(p0 - p1)