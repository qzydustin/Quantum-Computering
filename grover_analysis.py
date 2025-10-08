#!/usr/bin/env python3
"""
Run the same Grover(3 qubits, marked 111) on:
- Ideal (noiseless)
- Real device (IBM Quantum)
- Noisy simulator (derived from the same real backend)

Then overlay the bitstring distributions on one figure for comparison.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import json

from program.grover_algorithm import grover_algorithm
from ibm_quantum_connector import QuantumServiceManager
from quantum_executor import QuantumExecutor
from qiskit import qpy
 


def bitstrings(n_qubits: int):
    return [format(i, f"0{n_qubits}b") for i in range(2**n_qubits)]


def probs_from_counts(counts: dict, shots: int, states: list[str]):
    if not counts or shots <= 0:
        return [0.0] * len(states)
    return [counts.get(s, 0) / shots for s in states]


def run_suite(exec_with_service: QuantumExecutor, qc, shots: int, states: list[str], level: int, save_prefix: str):
    backend_name = exec_with_service.backend.name
    # 1) Compile on real backend with given optimization level and persist ISA
    isa_path = os.path.join("archive", f"{save_prefix}_isa.qpy")
    isa_meta_path = os.path.splitext(isa_path)[0] + "_meta.json"
    res_real_compile = exec_with_service.run_circuit(
        qc,
        execution_type="real_device",
        shots=shots,
        optimization_level=level,
        save_isa_path=isa_path,
    )
    with open(isa_meta_path, "r") as f:
        meta = json.load(f)
    if meta.get("backend_name") != backend_name:
        raise RuntimeError(
            f"Suite L{level}: ISA backend ({meta.get('backend_name')}) != current backend ({backend_name})"
        )
    with open(isa_path, "rb") as f:
        loaded = list(qpy.load(f))
    if not loaded:
        raise RuntimeError(f"Suite L{level}: ISA QPY empty or corrupted")
    isa_circ = loaded[0]

    # 2) Ideal via executor.run_ideal_by_isa (noiseless, preserve real-device ISA layout)
    ideal_res = exec_with_service.run_ideal_by_isa(isa_circ, shots=shots)
    counts_ideal = ideal_res.get("counts", {}) or {}
    ideal_probs = probs_from_counts(counts_ideal, ideal_res.get("shots", shots), states)
    ideal_label = f"Ideal (L{level} ISA)"

    # 3) Real (reuse the compilation run results; no second submission)
    counts_real = res_real_compile.get("counts", {}) or {}
    real_probs = probs_from_counts(counts_real, shots, states)
    job_id = res_real_compile.get("job_id", "N/A")
    real_label = f"Real (L{level} ISA, {backend_name}, job={job_id})"

    # 4) Noisy via ISA
    noisy_res = exec_with_service.run_noisy_by_isa(
        real_backend=exec_with_service.backend,
        shots=shots,
        isa_circuit=isa_circ,
    )
    noisy_probs = probs_from_counts(noisy_res.get("counts"), noisy_res.get("shots", shots), states)
    noisy_label = f"Noisy (L{level} ISA, like {noisy_res.get('backend_like', backend_name)})"

    return {
        "ideal_probs": ideal_probs,
        "ideal_label": ideal_label,
        "real_probs": real_probs,
        "real_label": real_label,
        "noisy_probs": noisy_probs,
        "noisy_label": noisy_label,
    }


def main():
    n_qubits = 3
    # mark last state
    marked = [2**n_qubits - 1]
    shots = 1024
    states = bitstrings(n_qubits)

    qc = grover_algorithm(n_qubits=n_qubits, marked_states=marked)
    qc.name = f"Grover_{n_qubits}q_mark_last"

    # Connect IBM Quantum (required for real/noisy in this script)
    service_manager = QuantumServiceManager()
    if not service_manager.connect():
        print("Could not connect to IBM Quantum. Exiting.")
        return

    # Use the configured backend for both real and noisy paths
    exec_with_service = QuantumExecutor(service=service_manager.service)

    # Run two suites: level 0 and level 3
    try:
        base = run_suite(exec_with_service, qc, shots, states, level=0, save_prefix="grover_l0")
    except Exception as e:
        base = {
            "ideal_probs": [0.0]*len(states),
            "ideal_label": f"Ideal (L0 failed: {e})",
            "real_probs": [0.0]*len(states),
            "real_label": f"Real (L0 failed: {e})",
            "noisy_probs": [0.0]*len(states),
            "noisy_label": f"Noisy (L0 failed: {e})",
        }
    try:
        opt = run_suite(exec_with_service, qc, shots, states, level=3, save_prefix="grover_l3")
    except Exception as e:
        opt = {
            "ideal_probs": [0.0]*len(states),
            "ideal_label": f"Ideal (L3 failed: {e})",
            "real_probs": [0.0]*len(states),
            "real_label": f"Real (L3 failed: {e})",
            "noisy_probs": [0.0]*len(states),
            "noisy_label": f"Noisy (L3 failed: {e})",
        }

    # Plot: overlay distributions with aligned x positions (no jitter)
    x = np.arange(len(states))

    plt.figure(figsize=(12, 6))
    # base (L0)
    plt.plot(x, base["ideal_probs"], "-o", label=base["ideal_label"], linewidth=1.8, markersize=5, alpha=0.95, zorder=7)
    plt.plot(x, base["real_probs"], "-s", label=base["real_label"], linewidth=1.8, markersize=5, alpha=0.95, zorder=6)
    plt.plot(x, base["noisy_probs"], "-^", label=base["noisy_label"], linewidth=1.8, markersize=5, alpha=0.95, zorder=5)
    # optimized (L3, dashed)
    plt.plot(x, opt["ideal_probs"], "--o", label=opt["ideal_label"], linewidth=1.6, markersize=4, alpha=0.9, zorder=3)
    plt.plot(x, opt["real_probs"], "--s", label=opt["real_label"], linewidth=1.6, markersize=4, alpha=0.9, zorder=2)
    plt.plot(x, opt["noisy_probs"], "--^", label=opt["noisy_label"], linewidth=1.6, markersize=4, alpha=0.9, zorder=1)

    plt.xticks(x, states)
    plt.ylabel("Probability")
    plt.xlabel("Bitstring")
    plt.title(f"New Grover {n_qubits}-qubit (marked last) — L0 vs L3 ISA (Ideal/Real/Noisy)")
    plt.grid(axis="both", alpha=0.3, linestyle="--", linewidth=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"New Grover_{n_qubits}q_mark_last_00.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()