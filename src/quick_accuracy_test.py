import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer

from src.utils.load_data import load_data
from src.utils.paths import load_paths
from src.IQL.models.fixed_memory_iqc import FixedMemoryIQC
from src.IQL.backends.exact import ExactBackend
from src.IQL.backends.hardware_native import HardwareNativeBackend

NUM_SAMPLES = 30
SHOTS = 50
USE_EXACT_BACKEND = False

BASE_ROOT, PATHS = load_paths()
IQL_MODEL_PATH = os.path.join(BASE_ROOT, "results", "fixed_memory_iqc_sweep", "models", "fixed_memory_iqc_k2.pkl")

print(f"Loading {NUM_SAMPLES} test samples from patch chemelion dataset...")
_, X_test, _, y_test_pol = load_data(y="polar")
X_test = X_test[:NUM_SAMPLES]
y_test = y_test_pol[:NUM_SAMPLES]

normalizer = Normalizer(norm='l2')
X_test_norm = normalizer.fit_transform(X_test)

print(f"Loading model from {IQL_MODEL_PATH}...")
model = FixedMemoryIQC.load(IQL_MODEL_PATH)

if USE_EXACT_BACKEND:
    print("Using ExactBackend...")
    backend = ExactBackend()
else:
    print(f"Using HardwareNativeBackend with {SHOTS} shots...")
    backend = HardwareNativeBackend(shots=SHOTS)

for cs in model.memory_bank.class_states:
    cs.backend = backend

y_pred = model.predict(X_test_norm)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f} ({int(accuracy * NUM_SAMPLES)}/{NUM_SAMPLES})")
