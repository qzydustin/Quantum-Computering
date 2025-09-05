from __future__ import annotations
from typing import Dict, Any, Optional, Sequence
import json
import numpy as np

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.primitives import BackendSamplerV2  # 非 Runtime 用这个   [oai_citation:6‡IBM Quantum](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.primitives.BackendSamplerV2?utm_source=chatgpt.com)
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler  # Aer 的 SamplerV2   [oai_citation:7‡Qiskit](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.primitives.SamplerV2.html?utm_source=chatgpt.com)
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import Session, SamplerV2 as RuntimeSampler

# 如需 Runtime，请取消下一行注释，并在 __init__ 里设置 self.runtime_svc / self.runtime_backend
# from qiskit_ibm_runtime import SamplerV2 as RuntimeSampler  # Runtime 的 SamplerV2 (mode=...)   [oai_citation:8‡IBM Quantum](https://quantum.cloud.ibm.com/docs/api/qiskit-ibm-runtime/sampler-v2?utm_source=chatgpt.com)


class QuantumExecutor:
    """Quantum Circuit Executor - Core Execution Engine"""

    def __init__(self, service=None, config_file="config/quantum_config.json"):
        with open(config_file, "r") as f:
            self.config = json.load(f)

        self.service = service
        self.ideal_sim = AerSimulator()
        self.ideal_sampler = AerSampler()  # 可用于理想模拟（不强制）
        self.current_job = None

        if service:
            # 按配置选定一个 IBM 后端
            self.backend = self.service.backend(self.config["ibm_quantum"]["backend"])
            # Runtime 版本（可选）：self.sampler = RuntimeSampler(mode=self.backend)   [oai_citation:9‡IBM Quantum](https://quantum.cloud.ibm.com/docs/api/qiskit-ibm-runtime/sampler-v2?utm_source=chatgpt.com)
            self.sampler = BackendSamplerV2(backend=self.backend)  # 非 Runtime 版本   [oai_citation:10‡IBM Quantum](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.primitives.BackendSamplerV2?utm_source=chatgpt.com)
        else:
            self.backend = None
            self.sampler = None

    # ---------------------- 工具方法 ----------------------

    def _normalize_circuit(self, circ: QuantumCircuit) -> QuantumCircuit:
        """确保有测量；不在此处随机绑定参数（参数由 PUB 传入，三条路径共用）。"""
        qc = circ.copy()
        if not qc.clbits:
            qc.measure_all()
        return qc

    def _compile_for_target(self, circ: QuantumCircuit, target_backend) -> QuantumCircuit:
        # layout = { circ.qubits[i]: phys for i, phys in enumerate([11,12,10]) }
        # 使用预设编译器（不强制初始layout，适配任意量子比特数）
        pm = generate_preset_pass_manager(
            optimization_level=0, backend=target_backend
        )  # 预设编译器   [oai_citation:11‡IBM Quantum Documentation](https://docs.quantum.ibm.com/api/qiskit/0.42/qiskit.transpiler.preset_passmanagers.generate_preset_pass_manager?utm_source=chatgpt.com)
        return pm.run(circ)

    @staticmethod
    def _result_to_counts(result) -> Dict[str, int]:
        """v2：单 pub 合并计数"""
        return result[0].join_data().get_counts()  # 合并寄存器计数   [oai_citation:12‡Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com/questions/40735/getting-combined-counts-when-using-qiskit-ibm-runtime-samplerv2?utm_source=chatgpt.com)

    def _print_layout(self, isa_circ, title: str):
        tl = getattr(isa_circ, "layout", None)
        if tl is None:
            print(f"=== {title}: No TranspileLayout (no routing/full connectivity) ===")
            return
        idx_map = tl.final_index_layout()
        print(f"=== {title}: Final layout (virtual index -> output index) ===")
        for v_idx, out_idx in enumerate(idx_map):
            print(f"  q[{v_idx}] -> {out_idx}")
    # ---------------------- 统一入口 ----------------------

    def run_circuit(
        self,
        circuit: QuantumCircuit,
        execution_type: str,
        shots: Optional[int] = None,
        param_vals: Optional[Sequence[float]] = None,  # 三条路径共用同一份参数
    ) -> Dict[str, Any]:
        shots = shots or self.config["execution"]["shots"]
        if execution_type == "real_device":
            return self.run_real(circuit, self.backend, shots, param_vals)
        if execution_type == "ideal_simulator":
            return self.run_ideal(circuit, shots, param_vals)
        if execution_type == "noisy_simulator":
            return self.run_noisy(circuit, self.backend, shots, param_vals)
        return {"success": False, "error": f"Unknown execution_type: {execution_type}"}

    # ---------------------- 1) 理想模拟 ----------------------

    def run_ideal(
        self, circ: QuantumCircuit, shots: int = 1024, param_vals: Optional[Sequence[float]] = None
    ) -> Dict[str, Any]:
        sim = self.ideal_sim  # 全连通、无噪声；可能不会生成 layout（属正常）   [oai_citation:13‡Qiskit](https://qiskit.github.io/qiskit-aer/tutorials/1_aersimulator.html?utm_source=chatgpt.com)
        qc = self._normalize_circuit(circ)
        isa = self._compile_for_target(qc, sim)
        self._print_layout(isa, "ideal simulator")

        sampler = AerSampler.from_backend(sim)  # 编译端=执行端   [oai_citation:14‡Qiskit](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.primitives.SamplerV2.html?utm_source=chatgpt.com)
        pub = (isa, [] if not isa.parameters else (param_vals or [0.0]*len(isa.parameters)), shots)
        job = sampler.run([pub])  # PUB 形式   [oai_citation:15‡Qiskit](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.primitives.SamplerV2.html?utm_source=chatgpt.com)
        res = job.result()
        counts = self._result_to_counts(res)
        probs = {s: c / shots for s, c in counts.items()}
        return {"success": True, "execution_type": "ideal_simulator", "counts": counts,
                "probabilities": probs, "shots": shots, "method": "Aer SamplerV2"}

    # ---------------------- 2) 噪声模拟（真机派生） ----------------------

    def run_noisy(
        self, circ: QuantumCircuit, real_backend, shots: int = 1024,
        param_vals: Optional[Sequence[float]] = None
    ) -> Dict[str, Any]:
        # 1) 继承门集/耦合图
        sim = AerSimulator.from_backend(real_backend)  # 继承 basis_gates & coupling_map

        # 2) 显式从设备属性生成噪声模型并挂载到模拟器
        noise = NoiseModel.from_backend(real_backend)
        sim.set_options(noise_model=noise)

        # 3) 规范化 & 编译到同一个目标（谁编译谁执行）
        qc  = self._normalize_circuit(circ)
        isa = self._compile_for_target(qc, sim)
        self._print_layout(isa, f"noisy simulator (mimic {real_backend.name})")

        # 4) 用 Aer 的 SamplerV2 按 PUB 运行（V2 统一规范）
        sampler = AerSampler.from_backend(sim)
        pub = (isa, [] if not isa.parameters else (param_vals or [0.0]*len(isa.parameters)), shots)
        job = sampler.run([pub])
        res = job.result()

        counts = self._result_to_counts(res)
        probs  = {s: c / shots for s, c in counts.items()}
        return {
            "success": True,
            "execution_type": "noisy_simulator",
            "backend_like": real_backend.name,
            "counts": counts,
            "probabilities": probs,
            "shots": shots,
            "method": "Aer SamplerV2 (from_backend + explicit NoiseModel)"
        }

    # ---------------------- 3) 真机（或任意 BackendV2） ----------------------

    def run_real(
        self, circ: QuantumCircuit, backend, shots: int = 1024, param_vals: Optional[Sequence[float]] = None
    ) -> Dict[str, Any]:
        qc  = self._normalize_circuit(circ)
        isa = self._compile_for_target(qc, backend)
        self._print_layout(isa, f"real device: {backend.name}")

        # ✅ Open Plan：用“job mode”，直接把 backend 作为 mode 传入 Runtime SamplerV2
        sampler = RuntimeSampler(mode=backend)  # 不要用 Session；Open Plan 不支持。 [oai_citation:4‡IBM Quantum](https://quantum.cloud.ibm.com/docs/guides/execution-modes?utm_source=chatgpt.com)

        # 按 v2 规范使用 PUB（circuit, params?, shots?）
        pub = (isa, [] if not isa.parameters else (param_vals or [0.0]*len(isa.parameters)), shots)
        job = sampler.run([pub], shots=shots)
        res = job.result()

        counts = self._result_to_counts(res)
        return {
            "success": True,
            "execution_type": "real_device",
            "backend": backend.name,
            "job_id": job.job_id(),
            "counts": counts,
            "shots": shots,
            "method": "Runtime SamplerV2 (job mode)"
        }