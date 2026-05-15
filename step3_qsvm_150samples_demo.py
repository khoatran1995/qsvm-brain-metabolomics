import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from qiskit.circuit.library import PauliFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.state_fidelities import ComputeUncompute

USE_SIMULATOR = True # Set to False to run on real hardware!

if USE_SIMULATOR:
    from qiskit.primitives import StatevectorSampler
    print("Initializing local Quantum Simulator...")
    sampler = StatevectorSampler()
else:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit import transpile

    # 1. Save your account (Run this ONCE with your token, then you can comment it out again)
    QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="YOUR_IBM_TOKEN", overwrite=True)

    # 2. Connect to the service
    service = QiskitRuntimeService()

    # 3. Get the least busy real quantum computer
    print("Connecting to IBM Quantum and finding the least busy backend...")
    backend = service.least_busy(simulator=False, operational=True)
    print(f"Connected to backend: {backend.name}")

    # Custom Sampler that transpiles circuits automatically for real hardware!
    class HardwareSampler(SamplerV2):
        def __init__(self, backend):
            super().__init__(mode=backend)
            self.backend = backend
            
        def run(self, pubs, **kwargs):
            transpiled_pubs = []
            for pub in pubs:
                circuit, values = pub
                # Transpile the circuit to match the hardware ISA
                transpiled_circuit = transpile(circuit, backend=self.backend, optimization_level=3)
                transpiled_pubs.append((transpiled_circuit, values))
            
            # Set shots to 100 to avoid exceeding the system limit!
            kwargs.setdefault('shots', 100)
            return super().run(transpiled_pubs, **kwargs)

    sampler = HardwareSampler(backend)

print("--- The Adjusted Metabolomics Sprint (150 Samples Scale Test) ---")

print("\n1. Loading Prepared Data")
print("Loading exact 150 samples and 5 features...")

# Load the data we prepared to save time and ensure reproducibility
data = np.load("data/patient_data_150.npz")
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
print(f"Feature dimension: {X_train.shape[1]} (Ready for 5 qubits)")

print("\n2. The SVM Showdown: Classical SVM vs. QSVM")
# --- Classical SVM ---
print("Training Classical Support Vector Machine (SVM)...")
classical_svc = SVC(kernel='rbf', random_state=42)
classical_svc.fit(X_train, y_train)
classical_pred = classical_svc.predict(X_test)
classical_acc = accuracy_score(y_test, classical_pred)
print(f"Classical SVM Test Accuracy: {classical_acc * 100:.2f}%")

# --- Quantum SVM ---
print("\nTraining Quantum Support Vector Machine (QSVM)...")
print("Using PauliFeatureMap (Y, Z) with 5 qubits...")

feature_map = PauliFeatureMap(feature_dimension=5, reps=1, paulis=['Y', 'Z'], parameter_prefix='feat')

# Using real hardware with the sampler we created at the top!
fidelity = ComputeUncompute(sampler=sampler)
qkernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)

qsvc = QSVC(quantum_kernel=qkernel)
qsvc.fit(X_train, y_train)
qsvm_pred = qsvc.predict(X_test)
qsvm_acc = accuracy_score(y_test, qsvm_pred)

print("\n==============================================")
print("             SHOWDOWN RESULTS                 ")
print("==============================================")
print(f"Classical SVM Accuracy: {classical_acc * 100:.2f}%")
print(f"Quantum SVM Accuracy:   {qsvm_acc * 100:.2f}%")
print("==============================================")
print("\nNote: With 160 training samples, this submitted ~25,600 circuits.")
