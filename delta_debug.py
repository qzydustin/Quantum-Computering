from __future__ import annotations

from typing import List, Dict, Any, Optional, Sequence, Tuple
from datetime import datetime
import json

from qiskit import QuantumCircuit


class QuantumDeltaDebugger:
    """
    Quantum Circuit Delta Debugger (based on DDMin)

    - Targets `QuantumCircuit` compiled to backend ISA
    - Measures "loss" by comparing ideal vs. noisy simulation of target state probabilities
    - Uses DDMin to find the minimal set of segments causing the maximum loss
    - Provides "prefix scan" to locate layers with sudden error rate spikes
    """

    def __init__(
        self,
        executor,
        target_states: Sequence[int],
        tolerance: float = 0.02,
        noisy_execution_type: str = "real_device",
    ) -> None:
        self.executor = executor
        self.target_states = list(target_states)
        self.tolerance = float(tolerance)
        self.noisy_execution_type = noisy_execution_type  # "real_device" or "noisy_simulator"

        self.original_circuit: Optional[QuantumCircuit] = None
        self.segments: List[Dict[str, Any]] = []
        self.logical_n_qubits: Optional[int] = None
        self.test_count: int = 0
        self.ddmin_log: List[Dict[str, Any]] = []
        self.evaluations: List[Dict[str, Any]] = []

    # ---------- helpers ----------
    def _ensure_measured(self, circ: QuantumCircuit) -> QuantumCircuit:
        # If measurements already exist, return directly (no need to append)
        if any(inst.operation.name == "measure" for inst in circ.data):
            return circ
        n_q = circ.num_qubits
        # Ensure enough classical bits; explicitly measure one-to-one (qubit i -> clbit i)
        if circ.num_clbits < n_q:
            from qiskit import ClassicalRegister
            circ = circ.copy()
            circ.add_register(ClassicalRegister(n_q - circ.num_clbits))
        for i in range(n_q):
            circ.measure(i, i)
        return circ

    def _adaptive_tol(self, shots: int, p: float) -> float:
        import math
        sigma = math.sqrt(max(p * (1 - p), 1e-9) / max(shots, 1))
        return max(self.tolerance, 2.0 * sigma)

    # ---------- segmentation ----------
    def extract_circuit_segments(self, circuit: QuantumCircuit) -> List[Dict[str, Any]]:
        segments: List[Dict[str, Any]] = []
        current_layer: List[Dict[str, Any]] = []
        for i, instruction in enumerate(circuit.data):
            if instruction.operation.name in ["measure", "barrier", "delay", "reset"]:
                continue
            current_layer.append(
                {
                    "instruction": instruction,
                    "index": i,
                    "operation": instruction.operation.name,
                    "qubits": [circuit.find_bit(q).index for q in instruction.qubits],
                    "params": getattr(instruction.operation, "params", []),
                }
            )
            if self._should_end_layer(current_layer, instruction):
                if current_layer:
                    segments.append(
                        {
                            "instructions": current_layer.copy(),
                            "layer_id": len(segments),
                            "description": self._describe_layer(current_layer),
                        }
                    )
                    current_layer = []
        if current_layer:
            segments.append(
                {
                    "instructions": current_layer.copy(),
                    "layer_id": len(segments),
                    "description": self._describe_layer(current_layer),
                }
            )
        return segments

    def _should_end_layer(self, current_layer: List[Dict[str, Any]], instruction) -> bool:
        if len(current_layer) >= 5:
            return True
        if instruction.operation.name in ["cx", "cz", "mcx", "ccx"]:
            return True
        # If a conditional gate is encountered, cut the layer to avoid crossing conditional boundaries
        if getattr(instruction.operation, "condition", None) is not None:
            return True
        return False

    def _describe_layer(self, layer: List[Dict[str, Any]]) -> str:
        op_counts: Dict[str, int] = {}
        for inst in layer:
            op = inst["operation"]
            op_counts[op] = op_counts.get(op, 0) + 1
        return ", ".join([f"{cnt}×{op}" for op, cnt in op_counts.items()]) or "(empty)"

    # ---------- circuit builders ----------
    def build_circuit_without_segments(
        self, original_circuit: QuantumCircuit, segments_to_exclude: Sequence[int]
    ) -> QuantumCircuit:
        try:
            new_circuit = original_circuit.copy_empty_like()
        except Exception:
            new_circuit = QuantumCircuit(original_circuit.num_qubits, original_circuit.num_clbits)

        excluded_indices = set()
        for seg_idx in segments_to_exclude:
            if 0 <= seg_idx < len(self.segments):
                for inst in self.segments[seg_idx]["instructions"]:
                    excluded_indices.add(inst["index"])

        # Extract original measurement mappings
        original_measurements = []
        for i, inst in enumerate(original_circuit.data):
            if inst.operation.name == "measure":
                q_idx = original_circuit.find_bit(inst.qubits[0]).index
                c_idx = original_circuit.find_bit(inst.clbits[0]).index
                original_measurements.append((q_idx, c_idx))

        for i, inst in enumerate(original_circuit.data):
            op = inst.operation
            qargs = inst.qubits
            # Skip measurement and non-execution instructions
            if getattr(op, "name", None) in ["measure", "barrier", "delay", "reset"]:
                continue
            if i in excluded_indices:
                continue
            try:
                q_indices = [original_circuit.find_bit(q).index for q in qargs]
                mapped_qargs = [new_circuit.qubits[idx] for idx in q_indices]
            except Exception:
                mapped_qargs = [new_circuit.qubits[j] for j in range(len(qargs))]
            new_circuit.append(op, mapped_qargs, [])

        # Apply measurements: preserve original mappings if available
        if original_measurements:
            for q_idx, c_idx in original_measurements:
                new_circuit.measure(q_idx, c_idx)
        elif self.logical_n_qubits is None:
            new_circuit = self._ensure_measured(new_circuit)
        else:
            n = self.logical_n_qubits
            if new_circuit.num_clbits < n:
                tmp = QuantumCircuit(new_circuit.num_qubits, n)
                for inst in new_circuit.data:
                    if getattr(inst.operation, "name", None) != "measure":
                        tmp.append(inst.operation, inst.qubits, inst.clbits)
                new_circuit = tmp
            for i in range(n):
                new_circuit.measure(i, i)
        return new_circuit

    def build_circuit_with_segments(
        self, original_circuit: QuantumCircuit, segments_to_include: Sequence[int]
    ) -> QuantumCircuit:
        include_set = set(segments_to_include)
        try:
            new_circuit = original_circuit.copy_empty_like()
        except Exception:
            new_circuit = QuantumCircuit(original_circuit.num_qubits, original_circuit.num_clbits)
        
        # Extract original measurement mappings
        original_measurements = []
        for i, inst in enumerate(original_circuit.data):
            if inst.operation.name == "measure":
                q_idx = original_circuit.find_bit(inst.qubits[0]).index
                c_idx = original_circuit.find_bit(inst.clbits[0]).index
                original_measurements.append((q_idx, c_idx))
        
        include_indices: set[int] = set()
        for seg_idx in include_set:
            if 0 <= seg_idx < len(self.segments):
                for inst in self.segments[seg_idx]["instructions"]:
                    include_indices.add(inst["index"])
        for i, inst in enumerate(original_circuit.data):
            op = inst.operation
            qargs = inst.qubits
            # Skip measurement and non-execution instructions
            if getattr(op, "name", None) in ["measure", "barrier", "delay", "reset"]:
                continue
            if i not in include_indices:
                continue
            try:
                q_indices = [original_circuit.find_bit(q).index for q in qargs]
                mapped_qargs = [new_circuit.qubits[idx] for idx in q_indices]
            except Exception:
                mapped_qargs = [new_circuit.qubits[j] for j in range(len(qargs))]
            new_circuit.append(op, mapped_qargs, [])
        
        # Apply measurements: preserve original mappings if available
        if original_measurements:
            for q_idx, c_idx in original_measurements:
                new_circuit.measure(q_idx, c_idx)
        elif self.logical_n_qubits is None:
            new_circuit = self._ensure_measured(new_circuit)
        else:
            n = self.logical_n_qubits
            if new_circuit.num_clbits < n:
                tmp = QuantumCircuit(new_circuit.num_qubits, n)
                for inst in new_circuit.data:
                    if getattr(inst.operation, "name", None) != "measure":
                        tmp.append(inst.operation, inst.qubits, inst.clbits)
                new_circuit = tmp
            for i in range(n):
                new_circuit.measure(i, i)
        return new_circuit

    # ---------- evaluation ----------
    def evaluate_circuit(self, circuit: QuantumCircuit, shots: int = 2048) -> Tuple[float, float, Dict[str, int]]:
        self.test_count += 1

        def _project_counts(counts: Dict[str, int], n_bits: int) -> Dict[str, int]:
            from collections import defaultdict
            acc = defaultdict(int)
            for k, v in (counts or {}).items():
                # Qiskit uses big-endian: remove spaces, take first n_bits (corresponding to lower-numbered qubits)
                bits = k.replace(" ", "")
                # Keep only the first n_bits (corresponding to qubits 0 to n_bits-1)
                if len(bits) >= n_bits:
                    acc[bits[:n_bits]] += v
            return dict(acc)

        n_data = self.logical_n_qubits or circuit.num_qubits

        ideal = self.executor.run_circuit(circuit, execution_type="ideal_simulator", shots=shots)
        if not ideal.get("success"):
            raise RuntimeError(f"Ideal simulation failed: {ideal.get('error')}")
        ideal_counts = _project_counts(ideal.get("counts", {}), n_data)
        total_ideal = max(1, sum(ideal_counts.values()))

        def _key(state: int) -> str:
            # Qiskit big-endian: generate standard binary string
            return format(state, f"0{n_data}b")

        ideal_target_prob = sum(ideal_counts.get(_key(s), 0) for s in self.target_states) / total_ideal

        noisy = self.executor.run_circuit(circuit, execution_type=self.noisy_execution_type, shots=shots)
        if not noisy.get("success"):
            # Retry once; if still fails, throw error to avoid treating failure as 0 probability
            noisy = self.executor.run_circuit(circuit, execution_type=self.noisy_execution_type, shots=shots)
        if not noisy.get("success"):
            raise RuntimeError(f"Noisy simulation failed: {noisy.get('error')}")
        noisy_counts = _project_counts(noisy.get("counts", {}), n_data)
        total_noisy = max(1, sum(noisy_counts.values()))
        noisy_target_prob = sum(noisy_counts.get(_key(s), 0) for s in self.target_states) / total_noisy

        return ideal_target_prob, noisy_target_prob, ideal.get("counts", {})

    # ---------- ddmin ----------
    def ddmin(self, baseline_loss: Optional[float] = None) -> List[int]:
        if self.original_circuit is None:
            return []
        shots = 2048
        full_circuit = self.build_circuit_without_segments(self.original_circuit, [])
        full_ideal, full_noisy, _ = self.evaluate_circuit(full_circuit, shots=shots)
        loss = full_ideal - full_noisy if baseline_loss is None else baseline_loss
        # Clear ddmin log
        self.ddmin_log.clear()

        candidates = list(range(len(self.segments)))
        n = 2
        while len(candidates) >= 2:
            subsets = self._split(candidates, n)
            progressed = False
            for subset in subsets:
                complement = [i for i in candidates if i not in subset]
                if not complement:
                    continue
                test_circ = self.build_circuit_without_segments(self.original_circuit, subset)
                t_ideal, t_noisy, _ = self.evaluate_circuit(test_circ, shots=shots)
                t_loss = t_ideal - t_noisy
                tol_t = self._adaptive_tol(shots, 0.5 * (t_ideal + t_noisy))
                # Record ddmin evaluation
                self.evaluations.append({
                    "mode": "ddmin",
                    "excluded": subset,
                    "included": None,
                    "shots": shots,
                    "ideal": t_ideal,
                    "noisy": t_noisy,
                    "loss": t_loss,
                })
                log_entry = {
                    "action": "test_subset",
                    "excluded": subset,
                    "kept": complement,
                    "ideal": t_ideal,
                    "noisy": t_noisy,
                    "loss": t_loss,
                    "tol": tol_t,
                    "prev_loss": loss,
                    "progressed": False,
                }
                if (loss - t_loss) > tol_t:
                    candidates = subset
                    loss = t_loss
                    log_entry["progressed"] = True
                    self.ddmin_log.append(log_entry)
                    progressed = True
                    break
                # Try testing complement
                comp_circ = self.build_circuit_without_segments(self.original_circuit, complement)
                c_ideal, c_noisy, _ = self.evaluate_circuit(comp_circ, shots=shots)
                c_loss = c_ideal - c_noisy
                tol_c = self._adaptive_tol(shots, 0.5 * (c_ideal + c_noisy))
                # Record ddmin evaluation (complement)
                self.evaluations.append({
                    "mode": "ddmin",
                    "excluded": complement,
                    "included": None,
                    "shots": shots,
                    "ideal": c_ideal,
                    "noisy": c_noisy,
                    "loss": c_loss,
                })
                log_entry_comp = {
                    "action": "test_complement",
                    "excluded": complement,
                    "kept": subset,
                    "ideal": c_ideal,
                    "noisy": c_noisy,
                    "loss": c_loss,
                    "tol": tol_c,
                    "prev_loss": loss,
                    "progressed": False,
                }
                if (loss - c_loss) > tol_c:
                    candidates = complement
                    loss = c_loss
                    log_entry_comp["progressed"] = True
                    self.ddmin_log.append(log_entry_comp)
                    progressed = True
                    break
                # Both did not progress, record both logs
                self.ddmin_log.append(log_entry)
                self.ddmin_log.append(log_entry_comp)
            if not progressed:
                if n < len(candidates):
                    n = min(n * 2, len(candidates))
                else:
                    break
        return candidates

    def _split(self, items: List[int], n: int) -> List[List[int]]:
        if n >= len(items):
            return [[x] for x in items]
        size = len(items) // n
        rem = len(items) % n
        out: List[List[int]] = []
        start = 0
        for i in range(n):
            end = start + size + (1 if i < rem else 0)
            if start < len(items):
                out.append(items[start:end])
            start = end
        return [s for s in out if s]

    # ---------- spike scan ----------
    def scan_loss_by_prefix(self, shots: int = 1024) -> Dict[str, Any]:
        if self.original_circuit is None:
            raise RuntimeError("original_circuit is not set")
        n_seg = len(self.segments)
        curve: List[Dict[str, Any]] = []
        for i in range(n_seg):
            include = list(range(i + 1))  # First i layers (inclusive)
            test = self.build_circuit_with_segments(self.original_circuit, include)
            p_ideal, p_noisy, _ = self.evaluate_circuit(test, shots=shots)
            loss = p_ideal - p_noisy
            curve.append({
                "layer_id": i,
                "included_segments": include,
                "ideal": p_ideal,
                "noisy": p_noisy,
                "loss": loss,
            })
            # Record prefix evaluation
            self.evaluations.append({
                "mode": "prefix",
                "excluded": None,
                "included": include,
                "shots": shots,
                "ideal": p_ideal,
                "noisy": p_noisy,
                "loss": loss,
            })
        # Find maximum increase (spike) in adjacent differences
        spike_idx = None
        spike_delta = 0.0
        for k in range(1, len(curve)):
            delta = curve[k]["loss"] - curve[k - 1]["loss"]
            if delta > spike_delta:
                spike_delta = delta
                spike_idx = k
        return {
            "curve": curve,  # [(layer_idx, loss)]
            "spike_index": spike_idx,  # None or first layer index with significant spike
            "spike_delta": spike_delta,
        }

    # ---------- public API ----------
    def debug_circuit(
        self,
        circuit: QuantumCircuit,
        n_qubits: int,
        target_states: Sequence[int],
    ) -> Dict[str, Any]:
        self.original_circuit = circuit
        self.target_states = list(target_states)
        self.logical_n_qubits = int(n_qubits)
        self.test_count = 0
        self.ddmin_log.clear()
        self.evaluations.clear()

        self.segments = self.extract_circuit_segments(circuit)

        # First get baseline loss for reporting
        full = self.build_circuit_without_segments(self.original_circuit, [])
        base_ideal, base_noisy, _ = self.evaluate_circuit(full)
        baseline_loss = base_ideal - base_noisy
        # Record baseline evaluation
        self.evaluations.append({
            "mode": "baseline",
            "excluded": [],
            "included": list(range(len(self.segments))),
            "shots": 2048,
            "ideal": base_ideal,
            "noisy": base_noisy,
            "loss": baseline_loss,
        })

        minimal_set = self.ddmin(baseline_loss)

        analysis = self.analyze_problematic_segments(minimal_set)
        meta = {
            "timestamp": datetime.now().isoformat(),
            "logical_n_qubits": self.logical_n_qubits,
            "bit_endianness": "big (Qiskit standard): bitstring[0] = qubit 0, bitstring[i] = qubit i",
            "evaluation": {
                "shots_ddmin": 2048,
                "tolerance_base": self.tolerance,
                "tolerance_mode": "adaptive_2sigma",
                "noisy_retry": 1,
                "noisy_execution_type": self.noisy_execution_type,
            },
        }

        return {
            "meta": meta,
            "total_segments": len(self.segments),
            "problematic_segments": minimal_set,
            "analysis": analysis,
            "baseline_loss": baseline_loss,
            "test_count": self.test_count,
            "segments_info": self.segments,
            "ddmin_log": self.ddmin_log,
            "evaluations": self.evaluations,
        }

    def analyze_problematic_segments(self, problematic_segments: Sequence[int]) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "segment_count": len(problematic_segments),
            "segments": [],
            "operation_analysis": {},
            "qubit_analysis": {},
        }
        ops: List[str] = []
        qubits: set[int] = set()
        for idx in problematic_segments:
            if 0 <= idx < len(self.segments):
                seg = self.segments[idx]
                info = {
                    "layer_id": seg["layer_id"],
                    "description": seg["description"],
                    "instructions": len(seg["instructions"]),
                    "operations": [],
                }
                for inst in seg["instructions"]:
                    info["operations"].append(
                        {
                            "operation": inst["operation"],
                            "qubits": inst["qubits"],
                            "params": inst["params"],
                        }
                    )
                    ops.append(inst["operation"])
                    qubits.update(inst["qubits"])
                out["segments"].append(info)
        from collections import Counter
        out["operation_analysis"] = dict(Counter(ops))
        out["qubit_analysis"] = {
            "affected_qubits": sorted(qubits),
            "qubit_count": len(qubits),
        }
        return out

    # ---------- report ----------
    def save_debug_report(self, result: Dict[str, Any], filename: Optional[str] = None) -> str:
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"delta_debug_report_{ts}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        return filename


def run_delta_debug_on_isa(
    executor,
    isa_circuit: QuantumCircuit,
    n_qubits: int,
    marked_states: Sequence[int],
    tolerance: float = 0.02,
    noisy_execution_type: str = "real_device",
) -> Dict[str, Any]:
    """
    Run Delta Debug analysis
    
    Args:
        executor: Quantum circuit executor
        isa_circuit: ISA-compiled quantum circuit
        n_qubits: Number of logical qubits
        marked_states: List of target states (integers)
        tolerance: Tolerance threshold
        noisy_execution_type: Noisy execution type, either "real_device" or "noisy_simulator"
    
    Returns:
        Debug result dictionary
    """
    dbg = QuantumDeltaDebugger(
        executor=executor,
        target_states=marked_states,
        tolerance=tolerance,
        noisy_execution_type=noisy_execution_type
    )
    result = dbg.debug_circuit(isa_circuit, n_qubits=n_qubits, target_states=marked_states)
    # Also perform prefix scan to detect spikes
    scan_shots = 512
    loss_scan = dbg.scan_loss_by_prefix(shots=scan_shots)
    result["loss_scan"] = loss_scan
    # Update prefix scan shots in meta
    try:
        result["meta"]["evaluation"]["shots_prefix_scan"] = scan_shots
    except Exception:
        pass
    # Automatically save report
    dbg.save_debug_report(result)
    return result


