#!/usr/bin/env python3
"""
TVD vs Depth experiment using Qiskit Aer.

For a family of circuits with gradually increasing depth, this script samples
each circuit on:
  1) An ideal (noise-free) Aer simulator
  2) A noisy Aer simulator using a noise model imported from the real backend `ibm_brisbane`

It then computes the Total Variation Distance (TVD) between the two
distributions to quantify how error grows with circuit depth.
"""

from __future__ import annotations
import csv
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from qiskit import QuantumCircuit, transpile
from qiskit.result import Counts
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel


def get_noise_model_via_config() -> Tuple[NoiseModel, List[str]]:
    """Build a noise model from a real IBM backend using credentials in config/quantum_config.json.

    Uses `ibm_quantum_connector.QuantumServiceManager` to read the config and connect.
    The backend is read from the same config file (key: ibm_quantum.backend).
    """
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from ibm_quantum_connector import QuantumServiceManager  # import after sys.path fix

    config_path = project_root / "config" / "quantum_config.json"
    manager = QuantumServiceManager(config_file=str(config_path))
    if not manager.connect():
        raise RuntimeError("Failed to connect to IBM Quantum service using config/quantum_config.json")
    backend = manager.select_backend()
    if backend is None:
        raise RuntimeError("Failed to select backend from config/quantum_config.json")
    noise_model = NoiseModel.from_backend(backend)
    basis_gates = list(noise_model.basis_gates)
    return noise_model, basis_gates


def get_executor_from_config():
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from ibm_quantum_connector import QuantumServiceManager
    from quantum_executor import QuantumExecutor

    config_path = project_root / "config" / "quantum_config.json"
    manager = QuantumServiceManager(config_file=str(config_path))
    if not manager.connect():
        raise RuntimeError("Failed to connect to IBM Quantum service using config/quantum_config.json")
    executor = QuantumExecutor(service=manager.service, config_file=str(config_path))
    return executor


def build_layered_circuit(num_qubits: int, num_layers: int, seed: int) -> QuantumCircuit:
    """Construct a circuit that increases in depth by stacking layers.

    Each layer applies parameterized single-qubit rotations and a simple
    entangling pattern (CX in a line), which grows depth roughly linearly with
    the number of layers.
    """
    rng = random.Random(seed)
    qc = QuantumCircuit(num_qubits, num_qubits)

    # Start with Hadamards to create a superposition baseline
    for q in range(num_qubits):
        qc.h(q)

    for _ in range(num_layers):
        # Single-qubit rotations with stable pseudo-random angles
        for q in range(num_qubits):
            theta = (rng.random() - 0.5) * math.pi  # [-pi/2, pi/2]
            phi = (rng.random() - 0.5) * math.pi
            lam = (rng.random() - 0.5) * math.pi
            qc.u(theta, phi, lam, q)

        # Linear entangling layer (nearest-neighbor CX)
        for q in range(0, num_qubits - 1, 2):
            qc.cx(q, q + 1)
        for q in range(1, num_qubits - 1, 2):
            qc.cx(q, q + 1)

    qc.barrier()
    qc.measure(range(num_qubits), range(num_qubits))
    return qc


def build_layered_circuit_type_b(num_qubits: int, num_layers: int, seed: int) -> QuantumCircuit:
    """Circuit type B: Alternating Rx/Rz with ring entanglement via CX.

    Per layer:
      - Apply Rx and Rz with distinct pseudo-random angles per qubit
      - Apply CX in a ring topology (q -> (q+1) mod N)
    """
    rng = random.Random(seed * 7919 + 17)  # derive a different stream from seed
    qc = QuantumCircuit(num_qubits, num_qubits)

    # Start with Rx(pi/2) to differ from type A's H layer
    for q in range(num_qubits):
        qc.rx(math.pi / 2, q)

    for _ in range(num_layers):
        for q in range(num_qubits):
            theta = (rng.random() - 0.5) * math.pi
            phi = (rng.random() - 0.5) * math.pi
            qc.rx(theta, q)
            qc.rz(phi, q)

        # Ring CX entanglers
        for q in range(num_qubits):
            qc.cx(q, (q + 1) % num_qubits)

    qc.barrier()
    qc.measure(range(num_qubits), range(num_qubits))
    return qc


def counts_to_probabilities(counts: Counts, num_shots: int) -> Dict[str, float]:
    return {bitstr: c / float(num_shots) for bitstr, c in counts.items()}


def total_variation_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p.keys()) | set(q.keys())
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


@dataclass
class ExperimentConfig:
    num_qubits: int
    max_layers: int
    layer_step: int
    shots: int
    seed: int
    plot: bool
    out_csv: Path | None


def run_experiment(cfg: ExperimentConfig) -> Dict[str, List[Tuple[int, float]]]:
    executor = get_executor_from_config()

    def _prob_from_counts(counts: Dict[str, int], shots: int) -> Dict[str, float]:
        total = float(max(1, shots))
        return {k: v / total for k, v in counts.items()}

    def _run_family(builder, fam_label: str) -> Dict[str, List[Tuple[int, float]]]:
        curves = {"sim": [], "real": []}
        for layers in range(0, cfg.max_layers + 1, cfg.layer_step):
            qc = builder(cfg.num_qubits, layers, seed=cfg.seed)

            ideal = executor.run_circuit(qc, execution_type="ideal_simulator", shots=cfg.shots)
            if not ideal.get("success"):
                raise RuntimeError(f"Ideal run failed: {ideal.get('error')}")
            p = _prob_from_counts(ideal["counts"], cfg.shots)

            noisy = executor.run_circuit(qc, execution_type="noisy_simulator", shots=cfg.shots)
            if noisy.get("success"):
                q_sim = _prob_from_counts(noisy["counts"], cfg.shots)
                print(p)
                print(q_sim)
                tvd_sim = total_variation_distance(p, q_sim)
                curves["sim"].append((layers, tvd_sim))
                print(f"circuit={fam_label}-sim  layers={layers:3d}  TVD={tvd_sim:.4f}")

            # real = executor.run_circuit(qc, execution_type="real_device", shots=cfg.shots)
            # if real.get("success"):
            #     q_real = _prob_from_counts(real["counts"], cfg.shots)
            #     tvd_real = total_variation_distance(p, q_real)
            #     curves["real"].append((layers, tvd_real))
            #     print(f"circuit={fam_label}-real layers={layers:3d}  TVD={tvd_real:.4f}")
            # else:
            #     print(f"circuit={fam_label}-real layers={layers:3d}  FAILED: {real.get('error')}")

        return {f"{fam_label}-sim": curves["sim"], f"{fam_label}-real": curves["real"]}

    results = {}
    results.update(_run_family(build_layered_circuit, "a"))
    results.update(_run_family(build_layered_circuit_type_b, "b"))
    return results


def maybe_write_csv(results_by_family: Dict[str, List[Tuple[int, float]]], out_csv: Path | None) -> None:
    if out_csv is None:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["circuit", "layers", "tvd"])
        for fam, rows in results_by_family.items():
            for layers, tvd in rows:
                writer.writerow([fam, layers, tvd])
    print(f"Saved CSV: {out_csv}")


def maybe_plot(
    results_by_family: Dict[str, List[Tuple[int, float]]],
    title: str,
    out_png: Path | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not installed; skipping plot.")
        return

    plt.figure(figsize=(7, 4))
    for fam, rows in sorted(results_by_family.items()):
        layers = [x for x, _ in rows]
        tvds = [y for _, y in rows]
        label = f"{fam.upper()}"
        plt.plot(layers, tvds, marker="o", label=label)

    plt.xlabel("Circuit depth (layers)")
    plt.ylabel("Total Variation Distance (TVD)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        try:
            import matplotlib.pyplot as plt  # type: ignore # reuse same module
            plt.savefig(out_png, dpi=300, bbox_inches="tight")
            print(f"Saved plot: {out_png}")
        except Exception as _:
            pass
    plt.show()


def main() -> int:
    # Defaults
    num_qubits = 2
    max_layers = 20
    layer_step = 5
    shots = 1024
    seed = 1234

    out_csv: Path | None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("archive")
    out_csv = out_dir / f"tvd_vs_depth_{ts}.csv"
    out_png = out_dir / f"tvd_vs_depth_{ts}.png"

    cfg = ExperimentConfig(
        num_qubits=num_qubits,
        max_layers=max_layers,
        layer_step=layer_step,
        shots=shots,
        seed=seed,
        plot=True,
        out_csv=out_csv,
    )

    print(
        f"Running TVD vs Depth: qubits={cfg.num_qubits}, max_layers={cfg.max_layers}, "
        f"step={cfg.layer_step}, shots={cfg.shots}, backend=ibm_brisbane, circuits=a-sim,a-real,b-sim,b-real"
    )

    results_by_family = run_experiment(cfg)
    maybe_write_csv(results_by_family, cfg.out_csv)
    if cfg.plot:
        title = (
            f"TVD vs depth (qubits={cfg.num_qubits}, shots={cfg.shots}, backend=ibm_brisbane)"
        )
        maybe_plot(results_by_family, title=title, out_png=out_png)
    return 0


if __name__ == "__main__":
    sys.exit(main())


