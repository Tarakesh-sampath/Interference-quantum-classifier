import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, Pauli
from qiskit_aer import AerSimulator
from src.utils.common import build_transition_unitary

# Initialize high-performance GPU simulator
BACKEND = AerSimulator(method='statevector', device='GPU')

def build(psi, chi):
    """
    ISDO Circuit B': Transition-based interference (CORRECT PHYSICAL IMPLEMENTATION)
    
    This circuit measures Re⟨χ|ψ⟩ using a controlled transition unitary.
    
    Circuit structure:
        Ancilla: |0⟩ ──H──●────H──M
                           │
        Data:    |ψ⟩ ─────U_χψ────
    
    Where U_χψ is the transition unitary: U_χψ |ψ⟩ = |χ⟩
    
    This produces LINEAR interference, not quadratic!
    """
    # Ensure complex128 for Qiskit compatibility
    psi = np.asarray(psi, dtype=np.complex128)
    chi = np.asarray(chi, dtype=np.complex128)

    n = int(np.log2(len(psi)))
    qc = QuantumCircuit(1 + n, 1)
    
    anc = 0
    data = list(range(1, n + 1))
    
    # Prepare |ψ⟩ on data qubits
    from qiskit.circuit.library import StatePreparation
    qc.append(StatePreparation(psi), data)
    
    # Hadamard on ancilla
    qc.h(anc)
    
    # Controlled transition unitary
    U_chi_psi = build_transition_unitary(psi, chi)
    qc.append(U_chi_psi.control(1), [anc] + data)
    
    # Final Hadamard
    qc.h(anc)
    
    # Measure ancilla
    qc.measure(anc, 0)
    
    return qc


def run(psi, chi):
    """
    GPU-accelerated evaluation of Re⟨χ|ψ⟩ using AerSimulator
    """
    # Inputs are assumed normalized at this stage
    qc = build(psi, chi)
    
    # Use Aer's internal expectation value for speed
    qc_no_meas = qc.remove_final_measurements(inplace=False)
    qc_no_meas.save_expectation_value(Pauli('Z'), [0], label='isdo')
    
    # Transpile for the backend to decompose complex instructions like StatePreparation
    qc_optimized = transpile(qc_no_meas, BACKEND)
    
    # Run on GPU backend
    result = BACKEND.run(qc_optimized).result()
    z_exp = result.data()['isdo'].real
    return z_exp


def verify(psi, chi):
    """
    Verify that the circuit correctly computes Re⟨χ|ψ⟩
    """
    # Normalize inputs
    psi = np.asarray(psi, dtype=np.complex128)
    chi = np.asarray(chi, dtype=np.complex128)
    psi = psi / np.linalg.norm(psi)
    chi = chi / np.linalg.norm(chi)
    
    # Expected value
    expected = np.real(np.vdot(chi, psi))
    
    # Circuit result
    measured = run(psi, chi)
    
    # Check
    is_correct = np.allclose(measured, expected, atol=1e-10)
    
    print(f"Expected:  {expected}")
    print(f"Measured:  {measured}")
    print(f"Correct:   {is_correct}")
    
    return is_correct