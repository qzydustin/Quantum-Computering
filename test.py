#!/usr/bin/env python3
"""
End-to-end test for Grover algorithm on:
- Ideal (noiseless) simulator
- Noisy simulator (noise model from real backend)
- Real device (IBM Quantum)

Requirements:
- config/quantum_config.json must contain valid IBM Quantum credentials and a backend name
  for noisy simulator and real device sections to run.
"""

from program.grover_algorithm import grover_algorithm
from ibm_quantum_connector import QuantumServiceManager
from quantum_executor import QuantumExecutor


def print_top_counts(title: str, counts: dict, shots: int, top_k: int = 5) -> None:
    print(f"\n{title}")
    if not counts:
        print("  No counts returned.")
        return
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
    for state, cnt in sorted_counts:
        print(f"  {state}: {cnt} ({cnt/shots:.3f})")


def run_ideal(qc, shots: int = 1024) -> None:
    print("\n=== Running ideal (noiseless) simulator ===")
    executor = QuantumExecutor()
    result = executor.run_circuit(qc, execution_type="ideal_simulator", shots=shots)
    if result.get("success"):
        print_top_counts("Ideal counts (top)", result["counts"], shots)
    else:
        print(f"Ideal simulation failed: {result.get('error')}")


def run_noisy_and_real(qc, shots: int = 1024) -> None:
    print("\nConnecting to IBM Quantum for noisy/real runs...")
    service_manager = QuantumServiceManager()
    if not service_manager.connect():
        print("Could not connect to IBM Quantum. Skipping noisy and real runs.")
        return

    # Executor with service will attach backend from config
    try:
        executor = QuantumExecutor(service=service_manager.service)
    except Exception as e:
        print(f"Could not initialize QuantumExecutor with service: {e}")
        print("Skipping noisy and real runs.")
        return

    # # Noisy simulator (noise model from backend)
    print("\n=== Running noisy simulator (backend-derived noise) ===")
    try:
        noisy = executor.run_circuit(qc, execution_type="noisy_simulator", shots=shots)
        if noisy.get("success"):
            print_top_counts("Noisy counts (top)", noisy["counts"], shots)
        else:
            print(f"Noisy simulation failed: {noisy.get('error')}")
    except Exception as e:
        print(f"Noisy simulation raised: {e}")

    # Real device
    print("\n=== Running on real device ===")
    try:
        real = executor.run_circuit(qc, execution_type="real_device", shots=shots)
        if real.get("success"):
            print_top_counts("Real-device counts (top)", real["counts"], shots)
            print(f"Backend: {real.get('backend')}, Job ID: {real.get('job_id')}")
        else:
            print(f"Real-device run failed: {real.get('error')}")
    except Exception as e:
        print(f"Real-device run raised: {e}")


def main():
    # Build a small Grover circuit: 3 qubits, mark state 111
    qc = grover_algorithm(n_qubits=3, marked_states=[7])
    qc.name = "Grover_3q_mark_111"

    # Ideal simulator (offline)
    run_ideal(qc, shots=1024)

    # # Noisy simulator and real device (require IBM service credentials)
    run_noisy_and_real(qc, shots=1024)


if __name__ == "__main__":
    main()


