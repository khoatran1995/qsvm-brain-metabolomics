import json
from qiskit_ibm_runtime import RuntimeDecoder

def load_result(filename):
    with open(filename, "r") as f:
        result = json.load(f, cls=RuntimeDecoder)
    return result

res1 = load_result("workloads_100/job-d83bnqo0bvlc73d38ds0-result.json")
pub0 = res1[0]
bit_array = pub0.data.meas

print(f"Type of bit_array: {type(bit_array)}")
print(f"Attributes of bit_array: {dir(bit_array)}")

# Let's try to get counts
try:
    print(f"Counts: {bit_array.get_counts()}")
except Exception as e:
    print(f"Could not get counts with get_counts(): {e}")

try:
    print(f"Int counts: {bit_array.get_int_counts()}")
except Exception as e:
    print(f"Could not get counts with get_int_counts(): {e}")
