import os
import json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import Normalizer

# Project components
from src.utils.paths import load_paths
from src.utils.seed import set_seed
from src.IQL.models.fixed_memory_iqc import FixedMemoryIQC
from src.IQL.backends.hardware_native import HardwareNativeBackend

def main():
    set_seed(42)
    BASE_ROOT, PATHS = load_paths()
    EMBED_DIR = PATHS["embeddings"]
    
    # Configuration
    K_val = 2
    shots = 50
    EVAL_SAMPLES = 600  # Following the sample size in the example script
    
    print(f"Generating ISDO matrices and metrics for K={K_val}, shots={shots}...")

    # Load data
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
    
    # Model Path (specifically requested)
    IQL_PATH = os.path.join(BASE_ROOT, "results", "fixed_memory_iqc_sweep", "models", f"fixed_memory_iqc_k{K_val}.pkl")
    
    print(f"Loading Model from {IQL_PATH}...")
    iql_model = FixedMemoryIQC.load(IQL_PATH)
    
    # Inject shots into HardwareNativeBackend
    print(f"Injecting HardwareNativeBackend with {shots} shots...")
    new_backend = HardwareNativeBackend(shots=shots)
    for cs in iql_model.memory_bank.class_states:
        cs.backend = new_backend
        
    # Generate Predictions
    print("Generating predictions...")
    y_pred = iql_model.predict(X_test_norm)
    y_pred = np.array(y_pred)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test_pol, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # 2. Classification Metrics (Accuracy, Precision, Recall, F1)
    acc = accuracy_score(y_test_pol, y_pred)
    # Using 'weighted' average to account for class imbalance if any, though usually binary
    precision, recall, f1, _ = precision_recall_fscore_support(y_test_pol, y_pred, average='binary')
    
    report = classification_report(y_test_pol, y_pred, target_names=["Class 0", "Class 1"])
    print("\nClassification Report:")
    print(report)
    print(f"Accuracy: {acc:.4f}")
    
    # 3. Correlation Matrix (State Overlaps)
    class_states = iql_model.memory_bank.class_states
    n_states = len(class_states)
    corr_matrix = np.zeros((n_states, n_states))
    labels = [f"Class{cs.label}_Idx{i}" for i, cs in enumerate(class_states)]
    
    print(f"\nComputing {n_states}x{n_states} correlation matrix (absolute squared overlaps)...")
    for i in range(n_states):
        for j in range(n_states):
            v_i = class_states[i].vector
            v_j = class_states[j].vector
            # Absolute squared overlap |<v_i|v_j>|^2
            overlap = np.abs(np.vdot(v_i, v_j))**2
            corr_matrix[i, j] = float(overlap)
            
    print("Correlation Matrix:")
    print(corr_matrix)
    
    # Save results
    RESULTS_DIR = os.path.join(BASE_ROOT, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    RESULTS_FILE = os.path.join(RESULTS_DIR, f"isdo_matrices_k{K_val}_s{shots}.json")
    
    results = {
        "config": {
            "K": K_val,
            "shots": shots,
            "samples": EVAL_SAMPLES,
            "model_path": IQL_PATH
        },
        "metrics": {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "classification_report": report
        },
        "confusion_matrix": cm.tolist(),
        "correlation_matrix": corr_matrix.tolist(),
        "state_labels": labels
    }
    
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nResults successfully saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
