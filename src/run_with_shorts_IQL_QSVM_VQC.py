import os
import json
import numpy as np
import time
import pickle
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer

# Qiskit components
from qiskit_machine_learning.algorithms import VQC, QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit.primitives import BackendSamplerV2
from qiskit_aer import AerSimulator

# Project components
from src.utils.paths import load_paths
from src.utils.seed import set_seed
from src.IQL.models.fixed_memory_iqc import FixedMemoryIQC
from src.IQL.backends.hardware_native import HardwareNativeBackend

def main():
    set_seed()
    BASE_ROOT, PATHS = load_paths()
    EMBED_DIR = PATHS["embeddings"]
    
    # Shot counts to compare
    SHOT_LIST = [10, 100, 512, 1024, 2048, 4096]
    
    # Subsample for faster evaluation (optional, but recommended for shots > 512)
    EVAL_SAMPLES = 600
    
    print("Loading embeddings...")
    X = np.load(os.path.join(EMBED_DIR, "val_embeddings.npy"))
    y = np.load(os.path.join(EMBED_DIR, "val_labels.npy"))
    test_idx = np.load(os.path.join(EMBED_DIR, "split_test_idx.npy"))
    
    X_test = X[test_idx][:EVAL_SAMPLES]
    y_test = y[test_idx][:EVAL_SAMPLES]
    
    # y_test_pol for IQL (uses -1, +1)
    y_test_pol = np.where(y_test == 0, -1, 1) if np.min(y_test) == 0 else y_test
    
    # L2 Normalization for Amplitude Encoding
    normalizer = Normalizer(norm='l2')
    X_test_norm = normalizer.fit_transform(X_test)
    
    # Pre-compute test statevectors for QSVM optimization
    from qiskit.quantum_info import Statevector
    print(f"Pre-computing {EVAL_SAMPLES} test statevectors...")
    sv_test_list = [Statevector(x) for x in X_test_norm]
    
    # Model Paths
    IQL_PATH = os.path.join(BASE_ROOT, "results", "fixed_memory_iqc_sweep", "models", "fixed_memory_iqc_k2.pkl")
    QSVM_PATH = os.path.join(BASE_ROOT, "results", "qsvm", "qsvm_amp_model.dill")
    VQC_PATH = os.path.join(BASE_ROOT, "results", "vqc_amp_simple", "vqc_amp_simple.dill")
    
    print(f"Loading Models...")
    # Load IQL
    iql_model = FixedMemoryIQC.load(IQL_PATH)
    
    # Load QSVM
    qsvm_model = QSVC.load(QSVM_PATH)
    
    # Load VQC
    vqc_model = VQC.load(VQC_PATH)
    
    results = {
        "shots": SHOT_LIST,
        "iql_acc": [],
        "qsvm_acc": [],
        "vqc_acc": []
    }
    
    for shots in SHOT_LIST:
        print(f"\nEvaluating with {shots} shots...")
        
        # --- 1. IQL Evaluation ---
        print(" Evaluating IQL...")
        # Inject shots into HardwareNativeBackend
        new_backend = HardwareNativeBackend(shots=shots)
        # Update each ClassState in the memory bank (deep update)
        for cs in iql_model.memory_bank.class_states:
            cs.backend = new_backend
        
        y_pred = iql_model.predict(X_test_norm)
        acc = accuracy_score(y_test_pol, y_pred)
        results["iql_acc"].append(acc)
        print(f"  IQL Accuracy: {acc:.4f}")
        
        # --- 2. VQC Evaluation ---
        print(" Evaluating VQC...")
        vqc_model.sampler = BackendSamplerV2(backend=AerSimulator(shots=shots))
        y_pred = vqc_model.predict(X_test_norm)
        acc = accuracy_score(y_test, y_pred)
        results["vqc_acc"].append(acc)
        print(f"  VQC Accuracy: {acc:.4f}")
        
        print(" Evaluating QSVM...")
        # Since FidelityQuantumKernel with RawFeatureVector fails on inverse()
        # when parameters are unbound, and AerSimulator crashes on large batches,
        # we compute the probabilities using Statevector and then sample.
        
        from qiskit.quantum_info import Statevector
        
        support_vectors = qsvm_model._BaseLibSVM__Xfit[qsvm_model.support_]
        n_test = len(X_test_norm)
        n_support = len(support_vectors)
        
        # Pre-compute support statevectors
        print(f"  Pre-computing {n_support} support statevectors...")
        sv_support = [Statevector(sj) for sj in support_vectors]
        
        # Build K matrix
        K = np.zeros((n_test, n_support))
        print(f"  Computing {n_test * n_support} overlaps and sampling {shots} shots...")
        
        for i in range(n_test):
            sv_test = sv_test_list[i]
            for j in range(n_support):
                prob = np.abs(sv_test.data @ sv_support[j].data.conj())**2
                # Clip to [0, 1] to avoid numerical precision errors in binomial distribution
                prob = np.clip(float(prob), 0.0, 1.0)
                # Sample from binomial distribution to simulate shots
                n0 = np.random.binomial(shots, prob)
                K[i, j] = n0 / shots
        
        # Decision function: f(x) = sum alpha_i y_i K(x, s_i) + b
        decision = K @ qsvm_model.dual_coef_.T + qsvm_model.intercept_
        y_pred = np.where(decision >= 0, 1, 0).flatten()
        
        acc = accuracy_score(y_test, y_pred)
        results["qsvm_acc"].append(acc)
        print(f"  QSVM Accuracy: {acc:.4f}")

    # Save results
    RESULTS_FILE = os.path.join(BASE_ROOT, "results", "shots_comparison_results.json")
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(SHOT_LIST, results["iql_acc"], 'o-', label='IQL (FixedMemoryIQC)', linewidth=2)
    plt.plot(SHOT_LIST, results["qsvm_acc"], 's-', label='QSVM (Amplitude)', linewidth=2)
    plt.plot(SHOT_LIST, results["vqc_acc"], '^-', label='VQC (Amplitude)', linewidth=2)
    
    plt.xscale('log')
    plt.xticks(SHOT_LIST, [str(s) for s in SHOT_LIST])
    plt.xlabel('Shot Count (log scale)')
    plt.ylabel('Test Accuracy')
    plt.title(f'Quantum Model Performance vs. Shot Count (N={EVAL_SAMPLES})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    PLOT_FILE = os.path.join(BASE_ROOT, "results", "shots_comparison.png")
    plt.savefig(PLOT_FILE)
    print(f"Plot saved to {PLOT_FILE}")
    
    # plt.show() # Commented out for non-interactive run

if __name__ == "__main__":
    main()
