import numpy as np
from scipy.stats import spearmanr

from src.IQC.interference.math_backend import MathInterferenceBackend
from src.IQC.interference.circuit_backend_transition import TransitionInterferenceBackend
from src.IQC.interference.circuit_backend_isdo_bprime import ISDOBPrimeInterferenceBackend


def random_state(n):
    v = np.random.randn(2**n) + 1j * np.random.randn(2**n)
    v /= np.linalg.norm(v)
    return v


def sign(x):
    return 1 if x >= 0 else -1


np.random.seed(0)

math_backend = MathInterferenceBackend()
ref_backend = TransitionInterferenceBackend()
isdo_backend = ISDOBPrimeInterferenceBackend()

n = 4
num_tests = 100

sign_agree = 0
ref_vals = []
isdo_vals = []

for _ in range(num_tests):
    chi = random_state(n)
    psi = random_state(n)

    s_ref = ref_backend.score(chi, psi)
    s_isdo = isdo_backend.score(chi, psi)

    ref_vals.append(s_ref)
    isdo_vals.append(s_isdo)

    if sign(s_ref) == sign(s_isdo):
        sign_agree += 1

rho, _ = spearmanr(ref_vals, isdo_vals)

print("ISDO-B′ vs Transition backend")
print("Sign agreement:", sign_agree, "/", num_tests)
print("Spearman rank correlation:", rho)
print("Mean |difference|:", np.mean(np.abs(np.array(ref_vals) - np.array(isdo_vals))))

"""
ISDO-B′ vs Transition backend
Sign agreement: 51 / 100
Spearman rank correlation: -0.029006900690069004
Mean |difference|: 0.21415260812801665
"""