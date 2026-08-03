"""qiskit-aer noise models for the IQC circuit-validation tier.

The circuit must be transpiled against `noise_model.basis_gates`, otherwise the
StatePreparation blocks never decompose into the gates the errors attach to and the
noise is a silent no-op (wired in HardwareNativeBackend via basis_gates=).
"""

from __future__ import annotations

from typing import Dict

BASIS_1Q = ("sx", "x", "rz", "id")
BASIS_2Q = ("cx",)


def depolarizing_noise(p1: float, p2: float):
    """1-qubit depolarizing p1 on single-qubit gates, 2-qubit p2 on cx."""
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    nm = NoiseModel()
    if p1 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), list(BASIS_1Q))
    if p2 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), list(BASIS_2Q))
    return nm


def thermal_noise(T1_us: float, T2_us: float, gate_ns: Dict[str, float]):
    """Thermal relaxation (T1/T2) per gate; 2-qubit error is the tensor of 1-qubit."""
    from qiskit_aer.noise import NoiseModel, thermal_relaxation_error

    T1 = T1_us * 1e3  # ns
    T2 = T2_us * 1e3
    assert T2 <= 2 * T1, "T2 must be <= 2*T1"
    nm = NoiseModel()
    for gate, t in gate_ns.items():
        if t <= 0:
            continue
        err1 = thermal_relaxation_error(T1, T2, t)
        if gate in ("cx",):
            err2 = err1.tensor(thermal_relaxation_error(T1, T2, t))
            nm.add_all_qubit_quantum_error(err2, [gate])
        else:
            nm.add_all_qubit_quantum_error(err1, [gate])
    return nm


def transpiled_depth(chi, psi, basis_gates):
    """Report transpiled depth L and 2q-gate count of the ISDO circuit (for the
    analytic (1-p)^L factor)."""
    from qiskit import transpile
    from src.IQL.backends.hardware_native import HardwareNativeBackend

    hb = HardwareNativeBackend(shots=1)
    qc = hb._build_circuit(chi, psi)
    tqc = transpile(qc, basis_gates=list(basis_gates), optimization_level=1)
    ops = tqc.count_ops()
    twoq = sum(v for k, v in ops.items() if k in ("cx", "cz", "ecr"))
    return {"depth": int(tqc.depth()), "num_2q": int(twoq), "ops": dict(ops)}
