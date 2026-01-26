import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation, UnitaryGate


def load_statevector(vec):
    """
    Create a Qiskit StatePreparation gate from a normalized vector.
    
    NOTE: This is for CONCEPTUAL/ORACLE model only (Circuit A)
    For physical implementation, use build_transition_unitary instead
    """
    vec = np.asarray(vec, dtype=np.complex128)
    norm = np.linalg.norm(vec)
    if not np.isclose(norm, 1.0, atol=1e-12):
        raise ValueError("Statevector must be normalized")
    return StatePreparation(vec)


def statevector_to_unitary(psi):
    """
    Convert a statevector to a unitary operator using Householder efficiency.
    Construct a Householder reflection U such that U |e1> = |psi>
    where e1 = [1, 0, ..., 0]^T.
    
    This is O(D^2) to build the matrix, compared to O(D^3) for Gram-Schmidt.
    """
    psi = np.asarray(psi, dtype=np.complex128)
    norm = np.linalg.norm(psi)
    if norm > 1e-15:
        psi = psi / norm
    
    dim = len(psi)
    e1 = np.zeros(dim, dtype=np.complex128)
    e1[0] = 1.0
    
    # Adjust phase to avoid numerical instability (choose phase to make w large)
    # We want to map phase * e1 to psi where phase has same angle as psi[0]
    # This ensures w = phase * e1 - psi is stable.
    angle = np.angle(psi[0]) if np.abs(psi[0]) > 1e-10 else 0.0
    phase = np.exp(1j * angle)
    
    target = phase * e1
    w = target - psi
    w_norm = np.linalg.norm(w)
    
    if w_norm < 1e-12:
        # psi is already phase * e1, so just return identity * phase
        return np.eye(dim, dtype=np.complex128) * phase
    
    v = w / w_norm
    # R = I - 2vv* maps target (phase * e1) to psi
    # R * phase * e1 = psi  => R * e1 = psi * phase*
    # To get U * e1 = psi, we need U = R * phase
    H = (np.eye(dim, dtype=np.complex128) - 2.0 * np.outer(v, v.conj())) * phase
    return H



def build_transition_unitary(psi, chi):
    """
    Build the transition unitary U_chi_psi = U_chi @ U_psi^dagger
    
    This is the KEY OPERATION for physically realizable ISDO (Circuit B').
    
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
    
    # Verify it works
    psi_normalized = np.asarray(psi, dtype=np.complex128)
    psi_normalized = psi_normalized / np.linalg.norm(psi_normalized)
    chi_normalized = np.asarray(chi, dtype=np.complex128)
    chi_normalized = chi_normalized / np.linalg.norm(chi_normalized)
    
    result = U_chi_psi @ psi_normalized
    if not np.allclose(result, chi_normalized, atol=1e-10):
        raise ValueError("Transition unitary does not correctly map |psi⟩ to |chi⟩")
    
    return UnitaryGate(U_chi_psi)


def build_chi_state(class0_protos, class1_protos):
    """
    Build |chi> = sum_k |phi_k^0> - sum_k |phi_k^1>, normalized
    
    This constructs the reference state for ISDO classification.
    """
    chi = np.zeros_like(class0_protos[0], dtype=np.float64)

    for p in class0_protos:
        chi += p
    for p in class1_protos:
        chi -= p

    chi /= np.linalg.norm(chi)
    return chi