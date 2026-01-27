from src.ISDO.circuits.transition_isdo import run as run_isdo_circuit
from .base import InterferenceBackend


class TransitionBackend(InterferenceBackend):
    """
    Physically realizable ISDO implementation using shared optimized ISDO circuits.
    
    This backend uses the hardware-optimized Householder reflections and 
    high-precision float64 logic from the ISDO module.
    """
    
    def score(self, chi, psi) -> float:
        """
        Calculates the interference score using the optimized ISDO quantum circuit.
        """
        # Call the shared ISDO routine
        return float(run_isdo_circuit(psi, chi))