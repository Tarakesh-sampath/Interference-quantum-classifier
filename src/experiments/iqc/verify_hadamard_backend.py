import numpy as np

from src.IQC.interference.math_backend import MathInterferenceBackend
from src.IQC.interference.circuit_backend_hadamard import HadamardInterferenceBackend


def random_state(n):
    v = np.random.randn(2**n) + 1j * np.random.randn(2**n)
    v /= np.linalg.norm(v)
    return v


def sign(x):
    return 1 if x >= 0 else -1


np.random.seed(0)

math_backend = MathInterferenceBackend()
had_backend = HadamardInterferenceBackend()

n = 3  # small, exact verification
num_tests = 50

sign_agree = 0
vals = []

for _ in range(num_tests):
    chi = random_state(n)
    psi = random_state(n)

    s_math = math_backend.score(chi, psi)
    s_had = had_backend.score(chi, psi)

    vals.append((s_math, s_had))

    if sign(s_math) == sign(s_had):
        sign_agree += 1

print("Sign agreement:", sign_agree, "/", num_tests)
print("Mean abs error:", np.mean([abs(a - b) for a, b in vals]))

## output
"""
Sign agreement: 50 / 50
Mean abs error: 6.399047958183246e-16
"""