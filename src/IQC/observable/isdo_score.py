import numpy as np

def isdo_score(chi: np.ndarray, psi: np.ndarray) -> float:
    """
    Linear interference score: Re <chi | psi>
    """
    return float(np.real(np.vdot(chi, psi)))
