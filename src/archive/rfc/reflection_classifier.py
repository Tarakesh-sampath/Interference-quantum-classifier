# Reflection-Fidelity Classifier

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli
from qiskit.circuit.library import StatePreparation, UnitaryGate
from ... utils.common import load_statevector


def reflection_operator(chi):
    """
    Build R_chi = I - 2|chi><chi|
    """
    dim = len(chi)
    proj = np.outer(chi, chi.conj())
    return np.eye(dim) - 2 * proj


def build_isdo_circuit_b(psi, chi):
    """
    ISDO Circuit B: Phase kickback via reflection
    """
    n = int(np.log2(len(psi)))
    qc = QuantumCircuit(1 + n, 1)

    anc = 0
    data = list(range(1, n + 1))

    # Prepare |psi>
    state_prep_psi = StatePreparation(psi)
    qc.append(state_prep_psi, data)

    # Hadamard ancilla
    qc.h(anc)

    # Controlled reflection
    R = UnitaryGate(reflection_operator(chi), label="R_chi")
    qc.append(R.control(1), [anc] + data)

    # Interference
    qc.h(anc)

    # Measure ancilla
    qc.measure(anc, 0)

    return qc


def run_isdo_circuit_b(psi, chi):
    """
    Exact ⟨Z⟩ extraction
    """
    qc = build_isdo_circuit_b(psi, chi)
    qc_no_meas = qc.remove_final_measurements(inplace=False)
    sv = Statevector.from_instruction(qc_no_meas)
    z_exp = sv.expectation_value(Pauli('Z'), [0]).real
    return z_exp