import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
exe = "./Oef18"  # path to your compiled C++ executable
multipliers = [1, 2, 4, 8, 16, 32, 64, 128, 256]
num_runs = 10  # how many times to repeat each multiplier for averaging

# --- Helper function to run the program and parse time ---
def run_test(multiplier):
    cmd = [exe, str(multiplier)]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = result.stdout + result.stderr
    match = re.search(r"tijd=([\d.]+)", output)
    if match:
        return float(match.group(1))
    else:
        print(f"⚠️  Could not parse output for multiplier={multiplier}:\n{output}")
        return None

# --- Collect data ---
results = {}
for m in multipliers:
    print(f"Running multiplier={m} ...")
    times = []
    for _ in range(num_runs):
        t = run_test(m)
        if t is not None:
            times.append(t)
    if times:
        avg_time = np.mean(times)
        results[m] = avg_time
        print(f"  → average time = {avg_time:.6f} sec")
    else:
        results[m] = np.nan

# --- Plot the results ---
x = list(results.keys())
y = [results[m] for m in x]

plt.figure(figsize=(8,5))
plt.plot(x, y, marker='o', linestyle='-', linewidth=2)
plt.xscale('log', base=2)
plt.xlabel("Multiplier (afstand tussen threads in bytes)")
plt.ylabel("Gemiddelde uitvoeringstijd (sec)")
plt.title("False Sharing Effect — Gemiddelde tijd vs. Multiplier")
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
