import numpy as np
from src.ISDO.observables.isdo import isdo_observable

def test_isdo_observable():
    # Define two state vectors
    psi = np.array([1, 1]) / np.sqrt(2)
    chi = np.array([1, -1]) / np.sqrt(2)
    
    # Mathematical expected value: Re<chi|psi>
    expected = np.real(np.vdot(chi, psi))
    
    # Circuit-based value (default behavior now)
    measured = isdo_observable(chi, psi)
    
    # Fallback/Real value
    fallback = isdo_observable(chi, psi, real=True)
    
    print(f"Psi: {psi}")
    print(f"Chi: {chi}")
    print(f"Expected (Math): {expected}")
    print(f"Measured (Circuit): {measured} | {measured>expected}")
    print(f"Fallback (Real): {fallback}")
    
    assert np.allclose(measured, expected, atol=1e-10), f"Circuit-based result {measured} != expected {expected}"
    assert np.allclose(fallback, expected, atol=1e-10), f"Fallback result {fallback} != expected {expected}"
    
    print("Verification successful!")

if __name__ == "__main__":
    try:
        test_isdo_observable()
    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()
