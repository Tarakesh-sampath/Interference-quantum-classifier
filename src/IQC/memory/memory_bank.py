class MemoryBank:
    def __init__(self, class_states, backend):
        self.class_states = class_states
        self.backend = backend

    def scores(self, psi):
        return [
            self.backend.score(cs.vector, psi)
            for cs in self.class_states
        ]

    def winner(self, psi):
        scores = self.scores(psi)
        #idx = int(max(range(len(scores)), key=lambda i: abs(scores[i])))
        idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        return idx, scores[idx]

    def add_memory(self, chi_vector):
        from ..states.class_state import ClassState
        self.class_states.append(ClassState(chi_vector))
