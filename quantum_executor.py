from __future__ import annotations
from typing import Dict, Any, Optional, Sequence
import json

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import SamplerV2 as RuntimeSampler

from ibm_quantum_connector import QuantumServiceManager


def _sort_bitstrings(d: Dict[str, Any], num_bits: Optional[int] = None) -> Dict[str, Any]:
    """
    Returns a new dict sorted by bitstring key from '000...0' to '111...1'.
    """
    if not d:
        return {}
    # Determine bit width if not given
    if num_bits is None:
        num_bits = max(len(str(k)) for k in d)
    sorted_keys = sorted(d.keys(), key=lambda s: int(s.zfill(num_bits), 2))
    return {k.zfill(num_bits): d[k] for k in sorted_keys}

class QuantumExecutor:
    """Quantum Circuit Executor - ONLY ISA execution preserved"""

    def __init__(self, config_file="quantum_config.json"):
        with open(config_file, "r") as f:
            self.config = json.load(f)
        self.ideal_sim = AerSimulator()
        self.noisy_sim = None
        # Select IBM backend per config
        svc = QuantumServiceManager(config_file=config_file)
        assert svc.connect(), 'Failed to connect to IBM Quantum service'
        self.backend = svc.select_backend()
        assert self.backend is not None, 'Failed to select backend'
        
        sim = AerSimulator.from_backend(self.backend)
        noise = NoiseModel.from_backend(self.backend)
        sim.set_options(noise_model=noise)
        self.noisy_sim = sim

    @staticmethod
    def _result_to_counts(result) -> Dict[str, int]:
        r0 = result[0]
        data = getattr(r0, "data", None)
        if data is not None:
            meas = getattr(data, "meas", None)
            if meas is not None:
                get_counts = getattr(meas, "get_counts", None)
                if callable(get_counts):
                    return get_counts()
            get_counts_direct = getattr(data, "get_counts", None)
            if callable(get_counts_direct):
                return get_counts_direct()
        join_data = getattr(r0, "join_data", None)
        if callable(join_data):
            jd = join_data()
            get_counts = getattr(jd, "get_counts", None)
            if callable(get_counts):
                return get_counts()
        return {}

    @staticmethod
    def _counts_to_probabilities(counts: Dict[str, int], shots: int) -> Dict[str, float]:
        # Avoid division by zero
        if not counts or shots == 0:
            return {}
        return {s: c / shots for s, c in counts.items()}

    def _standard_result(
        self,
        *,
        execution_type: str,
        backend_name: Optional[str],
        job_id: Optional[str],
        counts: Dict[str, int],
        probabilities: Dict[str, float],
        shots: int,
        method: str,
    ) -> Dict[str, Any]:
        # Sort bitstrings from '000...0' to '111...1'
        # Figure out number of bits by taking the longest key
        num_bits = 0
        for d in (counts, probabilities):
            if d:
                this_max = max(len(str(k)) for k in d)
                num_bits = max(num_bits, this_max)
        sorted_counts = _sort_bitstrings(counts, num_bits) if counts else {}
        sorted_probs = _sort_bitstrings(probabilities, num_bits) if probabilities else {}

        result = {
            "success": True,
            "execution_type": execution_type,
            "backend": backend_name,
            "job_id": job_id,
            "counts": sorted_counts,
            "probabilities": sorted_probs,
            "shots": shots,
            "method": method,
        }
        return result

    def run_ideal_by_isa(
        self,
        isa_circuit: QuantumCircuit,
        shots: int = 1024,
        param_vals: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        sim = self.ideal_sim
        isa = isa_circuit
        sampler = AerSampler.from_backend(sim)
        pub = (isa, [] if not isa.parameters else (param_vals or [0.0]*len(isa.parameters)), shots)
        job = sampler.run([pub])
        res = job.result()
        counts = self._result_to_counts(res)
        probs = self._counts_to_probabilities(counts, shots)
        return self._standard_result(
            execution_type="ideal_simulator",
            backend_name=None if self.backend is None else getattr(self.backend, "name", None),
            job_id=None,
            counts=counts,
            probabilities=probs,
            shots=shots,
            method="Aer SamplerV2"
        )

    def run_noisy_by_isa(
        self,
        isa_circuit: QuantumCircuit,
        shots: int = 1024,
        param_vals: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        sim = self.noisy_sim
        isa = isa_circuit
        sampler = AerSampler.from_backend(sim)
        pub = (isa, [] if not isa.parameters else (param_vals or [0.0]*len(isa.parameters)), shots)
        job = sampler.run([pub])
        res = job.result()
        counts = self._result_to_counts(res)
        probs = self._counts_to_probabilities(counts, shots)
        return self._standard_result(
            execution_type="noisy_simulator",
            backend_name=None if self.backend is None else getattr(self.backend, "name", None),
            job_id=None,
            counts=counts,
            probabilities=probs,
            shots=shots,
            method="Aer SamplerV2 (from_backend + explicit NoiseModel)"
        )

    def run_real_by_isa(
        self,
        isa_circuit: QuantumCircuit,
        shots: int = 1024,
        param_vals: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        sampler = RuntimeSampler(mode=self.backend)
        pub = (isa_circuit, [] if not isa_circuit.parameters else (param_vals or [0.0]*len(isa_circuit.parameters)), shots)
        job = sampler.run([pub], shots=shots)
        res = job.result()
        counts = self._result_to_counts(res)
        probs = self._counts_to_probabilities(counts, shots)
        return self._standard_result(
            execution_type="real_device",
            backend_name=None if self.backend is None else getattr(self.backend, "name", None),
            job_id=job.job_id() if hasattr(job, "job_id") and callable(job.job_id) else None,
            counts=counts,
            probabilities=probs,
            shots=shots,
            method="Runtime SamplerV2 (job mode, ISA only)"
        )
    
    def run_circuit(
        self,
        isa_circuit: Optional[QuantumCircuit] = None,
        execution_type: Optional[str] = None,
        shots: Optional[int] = None,
        param_vals: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        if execution_type is None:
            raise RuntimeError("execution_type is required")
        shots = shots or self.config["execution"]["shots"]
        if isa_circuit is None:
            raise RuntimeError("ISA circuit is required")
        if execution_type == "ideal_simulator":
            return self.run_ideal_by_isa(isa_circuit, shots, param_vals)
        elif execution_type == "noisy_simulator":
            return self.run_noisy_by_isa(isa_circuit, shots, param_vals)
        elif execution_type == "real_device":
            return self.run_real_by_isa(isa_circuit, shots, param_vals)
        else:
            return {"success": False, "error": f"Unknown execution_type: {execution_type}"}