import numpy as np

class MemoryBank:
    """
    Holds multiple learned class states |chi^(m)>.
    Selection is purely interference-based.
    """

    def __init__(self, class_states):
        self.class_states = class_states  # list[ClassState]

    def scores(self, psi):
        return [
            float(np.real(np.vdot(cs.vector, psi)))
            for cs in self.class_states
        ]

    def winner(self, psi):
        scores = self.scores(psi)
        idx = int(np.argmax(np.abs(scores)))
        return idx, scores[idx]
