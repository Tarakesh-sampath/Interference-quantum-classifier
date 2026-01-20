import numpy as np

from src.quantum.isdo.circuits.circuit_a_controlled_state import run_isdo_circuit_a
from src.archive.rfc.reflection_classifier import run_isdo_circuit_b
from src.utils.common import build_chi_state


# Dummy normalized vectors for sanity test
psi = np.random.randn(32)
psi /= np.linalg.norm(psi)

phi0 = [np.random.randn(32) for _ in range(3)]
phi1 = [np.random.randn(32) for _ in range(3)]
phi0 = [p / np.linalg.norm(p) for p in phi0]
phi1 = [p / np.linalg.norm(p) for p in phi1]

chi = build_chi_state(phi0, phi1)

za = run_isdo_circuit_a(psi, chi)
zb = run_isdo_circuit_b(psi, chi)

print("Circuit A ⟨Z⟩:", za)
print("Circuit B ⟨Z⟩:", zb)
print("Difference:", abs(za - zb))
