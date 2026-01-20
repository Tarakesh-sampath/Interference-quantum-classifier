import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli
from qiskit.circuit.library import UnitaryGate


def statevector_to_unitary(psi):
    """
    Convert a statevector to a unitary operator that creates it from |0...0⟩
    Uses Gram-Schmidt to complete the unitary matrix.
    
    This creates U_psi such that U_psi |0...0⟩ = |psi⟩
    """
    psi = np.asarray(psi, dtype=np.complex128)
    dim = len(psi)
    
    # Normalize
    psi = psi / np.linalg.norm(psi)
    
    # Create unitary matrix where first column is psi
    U = np.zeros((dim, dim), dtype=complex)
    U[:, 0] = psi
    
    # Complete to full unitary using Gram-Schmidt orthogonalization
    for i in range(1, dim):
        # Start with standard basis vector
        v = np.zeros(dim, dtype=complex)
        v[i] = 1.0
        
        # Orthogonalize against all previous columns
        for j in range(i):
            v -= np.vdot(U[:, j], v) * U[:, j]
        
        # Normalize and store
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-10:
            U[:, i] = v / v_norm
        else:
            # Use random vector if degenerate
            v = np.random.randn(dim) + 1j * np.random.randn(dim)
            for j in range(i):
                v -= np.vdot(U[:, j], v) * U[:, j]
            U[:, i] = v / np.linalg.norm(v)
    
    return U


def build_transition_unitary(psi, chi):
    """
    Build the transition unitary U_chi_psi = U_chi @ U_psi^dagger
    
    This unitary satisfies: U_chi_psi |psi⟩ = |chi⟩
    
    Args:
        psi: Source statevector
        chi: Target statevector
    
    Returns:
        UnitaryGate that implements the transition
    """
    # Build unitaries that prepare each state from |0...0⟩
    U_psi = statevector_to_unitary(psi)
    U_chi = statevector_to_unitary(chi)
    
    # Transition unitary: U_chi @ U_psi^dagger
    U_chi_psi = U_chi @ U_psi.conj().T
    
    # Verify it works (optional, for debugging)
    psi_normalized = np.asarray(psi, dtype=np.complex128)
    psi_normalized = psi_normalized / np.linalg.norm(psi_normalized)
    chi_normalized = np.asarray(chi, dtype=np.complex128)
    chi_normalized = chi_normalized / np.linalg.norm(chi_normalized)
    
    result = U_chi_psi @ psi_normalized
    assert np.allclose(result, chi_normalized, atol=1e-10), \
        "Transition unitary does not map |psi⟩ to |chi⟩"
    
    return UnitaryGate(U_chi_psi)


def build_isdo_circuit_b_prime(psi, chi):
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
    n = int(np.log2(len(psi)))
    qc = QuantumCircuit(1 + n, 1)
    
    anc = 0
    data = list(range(1, n + 1))
    
    # Prepare |ψ⟩ on data qubits (in practice, this comes from previous computation)
    # For simulation, we'll use state preparation
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


def run_isdo_circuit_b_prime(psi, chi):
    """
    Exact (statevector) evaluation of ⟨Z⟩ which gives Re⟨χ|ψ⟩
    
    This is the CORRECT physical implementation of ISDO.
    """
    qc = build_isdo_circuit_b_prime(psi, chi)
    qc_no_meas = qc.remove_final_measurements(inplace=False)
    sv = Statevector.from_instruction(qc_no_meas)
    z_exp = sv.expectation_value(Pauli('Z'), [0]).real
    return z_exp


def verify_isdo_b_prime(psi, chi):
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
    measured = run_isdo_circuit_b_prime(psi, chi)
    
    # Check
    is_correct = np.allclose(measured, expected, atol=1e-10)
    
    print(f"Expected:  {expected}")
    print(f"Measured:  {measured}")
    print(f"Correct:   {is_correct}")
    
    return is_correct