# src/quantum/observables/isdo.py
import numpy as np

def isdo_observable(chi, psi) -> float:
    """
    ISDO observable:
    Linear interference score Re⟨χ|ψ⟩
    """
    return float(np.real(np.vdot(chi, psi)))
