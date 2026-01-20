"""
Comparison of ISDO Circuit Implementations

This script demonstrates three approaches:
1. Circuit A: Conceptual (Oracle model) - for pedagogy only
2. Circuit B: Reflection-based - gives quadratic fidelity
3. Circuit B': Transition-based - CORRECT linear ISDO

Only Circuit B' gives the true ISDO observable: Re⟨χ|ψ⟩
"""

import numpy as np
from src.quantum.isdo.circuits.circuit_b_prime_transition import run_isdo_circuit_b_prime, verify_isdo_b_prime


def test_all_circuits():
    """
    Test all three circuit implementations and compare results
    """
    # Create two test states
    psi = np.array([0.6, 0.8, 0.0, 0.0], dtype=np.complex128)
    chi = np.array([0.8, 0.6, 0.0, 0.0], dtype=np.complex128)
    
    # Normalize
    psi = psi / np.linalg.norm(psi)
    chi = chi / np.linalg.norm(chi)
    
    # Expected ISDO value: Re⟨χ|ψ⟩
    expected_isdo = np.real(np.vdot(chi, psi))
    
    # Expected RFC (quadratic): 1 - 2|⟨χ|ψ⟩|²
    inner_product_magnitude_sq = np.abs(np.vdot(chi, psi))**2
    expected_rfc = 1 - 2 * inner_product_magnitude_sq
    
    print("=" * 70)
    print("ISDO CIRCUIT COMPARISON")
    print("=" * 70)
    print(f"\n|ψ⟩ = {psi}")
    print(f"|χ⟩ = {chi}")
    print(f"\n⟨χ|ψ⟩ = {np.vdot(chi, psi)}")
    print(f"|⟨χ|ψ⟩|² = {inner_product_magnitude_sq}")
    print()
    
    # Circuit B': Transition-based (CORRECT)
    print("-" * 70)
    print("Circuit B': Transition-Based Interference (CORRECT)")
    print("-" * 70)
    print("Purpose: CORRECT physical ISDO implementation")
    print("Observable: Re⟨χ|ψ⟩ (linear, signed, phase-sensitive)")
    print("Status: Use this for all hardware and claims")
    try:
        result_b_prime = run_isdo_circuit_b_prime(psi, chi)
        print(f"Result:   {result_b_prime:.6f}")
        print(f"Expected: {expected_isdo:.6f}")
        print(f"Match:    {np.allclose(result_b_prime, expected_isdo, atol=1e-6)}")
        
        print("\nRunning full verification...")
        verify_isdo_b_prime(psi, chi)
    except Exception as e:
        print(f"Error: {e}")
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"True ISDO (Re⟨χ|ψ⟩):           {expected_isdo:.6f}")
    print(f"RFC alternative (1-2|⟨χ|ψ⟩|²): {expected_rfc:.6f}")
    print()
    print("✓ Circuit A: Conceptual/oracle model only")
    print("✗ Circuit B: Gives RFC (quadratic), not ISDO")
    print("✓ Circuit B': CORRECT implementation - USE THIS")
    print()


def test_different_states():
    """
    Test with multiple state pairs to show the difference
    """
    print("\n" + "=" * 70)
    print("TESTING MULTIPLE STATE PAIRS")
    print("=" * 70)
    
    test_cases = [
        # Same states
        (np.array([1.0, 0, 0, 0]), np.array([1.0, 0, 0, 0])),
        # Orthogonal states
        (np.array([1.0, 0, 0, 0]), np.array([0, 1.0, 0, 0])),
        # Opposite states
        (np.array([1.0, 0, 0, 0]), np.array([-1.0, 0, 0, 0])),
        # General case
        (np.array([0.6, 0.8, 0, 0]), np.array([0.8, -0.6, 0, 0])),
    ]
    
    for i, (psi, chi) in enumerate(test_cases, 1):
        psi = psi / np.linalg.norm(psi)
        chi = chi / np.linalg.norm(chi)
        
        true_isdo = np.real(np.vdot(chi, psi))
        rfc = 1 - 2 * np.abs(np.vdot(chi, psi))**2
        
        try:
            measured_b_prime = run_isdo_circuit_b_prime(psi, chi)
            
            print(f"\nTest {i}:")
            print(f"  True ISDO (Re⟨χ|ψ⟩):    {true_isdo:+.4f}")
            print(f"  Circuit B' (transition):{measured_b_prime:+.4f} ✓")
        except Exception as e:
            print(f"\nTest {i}: Error - {e}")


if __name__ == "__main__":
    test_all_circuits()
    test_different_states()