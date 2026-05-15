import json
import numpy as np
from qiskit_ibm_runtime import RuntimeDecoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def load_result(filename):
    print(f"Loading {filename}...")
    with open(filename, "r") as f:
        result = json.load(f, cls=RuntimeDecoder)
    return result

# Load results
res1 = load_result("workloads_100/job-d83bnqo0bvlc73d38ds0-result.json")
res2 = load_result("workloads_100/job-d83bpcftjchs73bp09eg-result.json")

# Extract fidelity (probability of '00000')
def get_fidelities(result_obj):
    fidelities = []
    for pub in result_obj:
        counts = pub.data.meas.get_counts()
        zero_count = counts.get('00000', 0)
        # Total shots is usually 100 as we saw in the wrapper, or we can get it
        total_shots = pub.data.meas.num_shots
        fidelities.append(zero_count / total_shots)
    return np.array(fidelities)

fid1 = get_fidelities(res1)
fid2 = get_fidelities(res2)

print(f"Extracted {len(fid1)} fidelities for training")
print(f"Extracted {len(fid2)} fidelities for testing")

# Reconstruct 80x80 training matrix
# Order is likely upper triangle excluding diagonal
N = 80
K_train = np.eye(N) # Diagonal is 1.0

idx = 0
for i in range(N):
    for j in range(i + 1, N):
        val = fid1[idx]
        K_train[i, j] = val
        K_train[j, i] = val # Symmetric
        idx += 1

# Reconstruct 20x80 test matrix
K_test = np.zeros((20, 80))
idx = 0
for i in range(20):
    for j in range(80):
        K_test[i, j] = fid2[idx]
        idx += 1

# Load true labels to find support vectors
data = np.load("patient_data_100.npz")
y_train = data['y_train']
y_test = data['y_test']
X_train = data['X_train']

print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Train classical SVC with precomputed kernel
svc = SVC(kernel='precomputed')
svc.fit(K_train, y_train)

print(f"\nSuccess! Found {len(svc.support_)} support vectors.")
print(f"Support Vector Indices: {svc.support_}")

# Let's print the features of the first few support vectors
print("\nFeatures of the first 3 Support Vectors (Angles):")
for i in range(min(3, len(svc.support_))):
    idx = svc.support_[i]
    print(f"Patient {idx}: {X_train[idx]}")

# Check accuracy just to see
y_pred = svc.predict(K_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy on Test Set using extracted kernel: {acc * 100:.2f}%")
