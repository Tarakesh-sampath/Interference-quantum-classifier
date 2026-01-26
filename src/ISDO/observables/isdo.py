# src/ISDO/observables/isdo.py
import numpy as np
from src.ISDO.circuits.transition_isdo import run as run_isdo_circuit

def isdo_observable(chi, psi, real=False) -> float:
    """
    ISDO observable:
    Linear interference score Re⟨χ|ψ⟩
    """
    if real:
        return float(np.real(np.vdot(chi, psi)))
    else:
        # Use the quantum circuit to compute the observable
        return run_isdo_circuit(psi, chi)

