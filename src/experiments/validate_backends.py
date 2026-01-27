import numpy as np
from tqdm import tqdm

from src.IQC.interference.exact_backend import ExactBackend
from src.IQC.interference.transition_backend import TransitionBackend
from src.IQC.interference.oracle_backend import OracleBackend

NUM_TRIALS = 200

exact = ExactBackend()
transition = TransitionBackend()
hadamard = OracleBackend()

sign_match_et = 0
sign_match_eh = 0

err_et = []
err_eh = []

for _ in tqdm(range(NUM_TRIALS), desc="Validating backends"):
    chi = np.random.randn(32)
    psi = np.random.randn(32)

    chi /= np.linalg.norm(chi)
    psi /= np.linalg.norm(psi)

    s_exact = exact.score(chi, psi)
    s_trans = transition.score(chi, psi)
    s_hadam = hadamard.score(chi, psi)

    sign_match_et += int(np.sign(s_exact) == np.sign(s_trans))
    sign_match_eh += int(np.sign(s_exact) == np.sign(s_hadam))

    err_et.append(abs(s_exact - s_trans))
    err_eh.append(abs(s_exact - s_hadam))

print("\n=== Backend Validation ===")
print(f"Exact vs Transition sign agreement : {sign_match_et}/{NUM_TRIALS}")
print(f"Exact vs Hadamard   sign agreement : {sign_match_eh}/{NUM_TRIALS}")
print(f"Mean |Exact - Transition| : {np.mean(err_et):.3e}")
print(f"Mean |Exact - Hadamard|   : {np.mean(err_eh):.3e}")

## output 
"""
Validating backends: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [08:56<00:00,  2.68s/it]

=== Backend Validation ===
Exact vs Transition sign agreement : 200/200
Exact vs Hadamard   sign agreement : 200/200
Mean |Exact - Transition| : 1.991e-13
Mean |Exact - Hadamard|   : 2.001e-16
"""