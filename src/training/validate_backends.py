import numpy as np

from src.IQL.backends.exact import ExactBackend
from src.IQL.backends.hadamard import HadamardBackend
from src.IQL.backends.transition import TransitionBackend
from src.IQL.backends.hardware_native import HardwareNativeBackend


def random_state(n_qubits, seed=None):
    if seed is not None:
        np.random.seed(seed)
    dim = 2 ** n_qubits
    v = np.random.randn(dim) + 1j * np.random.randn(dim)
    return v / np.linalg.norm(v)


def run_backend_tests(n_qubits=3, n_tests=20):
    backends = {
        "Exact": ExactBackend(),
        "Hadamard": HadamardBackend(),
        "Transition": TransitionBackend(),
        "HardwareNative": HardwareNativeBackend(),
    }

    print(f"\nRunning backend tests with {n_qubits} qubits\n")

    # Fix χ
    chi = random_state(n_qubits, seed=42)

    scores = {name: [] for name in backends}

    for i in range(n_tests):
        psi = random_state(n_qubits, seed=100 + i)

        print(f"Test {i + 1}")
        for name, backend in backends.items():
            s = backend.score(chi, psi)
            scores[name].append(s)
            print(f"  {name:10s}: {s:+.6f}")
        print()

    # ----------------------------------------------------
    # Analysis
    # ----------------------------------------------------
    print("\n=== Backend Agreement Analysis ===\n")

    exact = np.array(scores["Exact"])

    for name in ["Hadamard", "Transition", "HardwareNative"]:
        diff = np.max(np.abs(exact - np.array(scores[name])))
        print(f"Max |Exact - {name}| = {diff:.2e}")

    # PrimeB: sign + ordering only
    primeb = np.array(scores["HardwareNative"])

    sign_match = np.mean(np.sign(primeb) == np.sign(exact))
    print(f"\nHardwareNative sign agreement with Exact: {sign_match * 100:.1f}%")

    # Rank correlation (ordering)
    exact_rank = np.argsort(exact)
    primeb_rank = np.argsort(primeb)
    rank_corr = np.corrcoef(exact_rank, primeb_rank)[0, 1]
    print(f"HardwareNative rank correlation with Exact: {rank_corr:.3f}")


if __name__ == "__main__":
    run_backend_tests(n_qubits=3, n_tests=20)


"""
Test 194
  Exact     : -0.224492
  Hadamard  : -0.224492
  Transition: -0.224492
  PrimeB    : +0.095676

Test 195
  Exact     : -0.028519
  Hadamard  : -0.028519
  Transition: -0.028519
  PrimeB    : -0.423231

Test 196
  Exact     : +0.203938
  Hadamard  : +0.203938
  Transition: +0.203938
  PrimeB    : -0.201812

Test 197
  Exact     : +0.143895
  Hadamard  : +0.143895
  Transition: +0.143895
  PrimeB    : +0.035991

Test 198
  Exact     : -0.111603
  Hadamard  : -0.111603
  Transition: -0.111603
  PrimeB    : -0.143718

Test 199
  Exact     : +0.164120
  Hadamard  : +0.164120
  Transition: +0.164120
  PrimeB    : +0.107708

Test 200
  Exact     : +0.145881
  Hadamard  : +0.145881
  Transition: +0.145881
  PrimeB    : -0.250643


=== Backend Agreement Analysis ===

Max |Exact - Hadamard| = 3.22e-15
Max |Exact - Transition| = 4.97e-14

PrimeB sign agreement with Exact: 52.5%
PrimeB rank correlation with Exact: -0.004
"""