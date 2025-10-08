#!/usr/bin/env python3
"""
Print IBM Quantum backend info for the last 30 days (UTC 00:00), saving JSON snapshots.

- For each of the last 30 days (including today), fetch backend properties at midnight UTC and save as a JSON file.
- Output directory: archive/backend_properties

Usage:
  python print_backend_info.py
"""

from __future__ import annotations

import datetime as dt
import sys
import json
import os
from typing import Any, List, Dict

from ibm_quantum_connector import QuantumServiceManager


def _safe_len(obj) -> int:
    try:
        return len(obj)
    except Exception:
        return 0


def _collect_top_level_readout_error(props):
    ro = getattr(props, "readout_error", None)
    if isinstance(ro, list):
        return ro
    return None


def _props_to_serializable_dict(props, backend_name: str, snapshot_iso: str) -> Dict[str, Any]:
    # Metadata
    out: Dict[str, Any] = {
        "backend_name": getattr(props, "backend_name", backend_name),
        "backend_version": getattr(props, "backend_version", None),
        "last_update_date": str(getattr(props, "last_update_date", None)),
        "snapshot_utc": snapshot_iso,
    }

    # Qubits
    qubits = getattr(props, "qubits", None) or []
    top_level_ro = _collect_top_level_readout_error(props)
    qubits_out: List[Dict[str, Any]] = []
    for qi, qitems in enumerate(qubits):
        entry: Dict[str, Any] = {}
        try:
            for item in qitems or []:
                name = getattr(item, "name", None)
                value = getattr(item, "value", None)
                unit = getattr(item, "unit", None)
                if name is None:
                    continue
                entry[name] = {"value": value, "unit": unit}
        except Exception:
            pass
        if top_level_ro is not None and qi < len(top_level_ro) and "readout_error" not in entry:
            entry["readout_error"] = {"value": top_level_ro[qi], "unit": None}
        qubits_out.append(entry)
    out["qubits"] = qubits_out

    # Gates
    gates = getattr(props, "gates", None) or []
    gates_out: List[Dict[str, Any]] = []
    for gate in gates:
        gname = getattr(gate, "gate", None)
        gqubits = list(getattr(gate, "qubits", None) or [])
        params = {}
        for p in getattr(gate, "parameters", None) or []:
            pname = getattr(p, "name", None)
            pval = getattr(p, "value", None)
            punit = getattr(p, "unit", None)
            if pname is not None:
                params[pname] = {"value": pval, "unit": punit}
        gates_out.append({"gate": gname, "qubits": gqubits, "parameters": params})
    out["gates"] = gates_out

    return out


def _iso_filename(dt_obj: dt.datetime) -> str:
    # ISO-like but filesystem friendly
    return dt_obj.strftime("%Y-%m-%dT%H-%M-%SZ")


def _save_snapshot_json(backend, when: dt.datetime, out_dir: str) -> str | None:
    # Fetch properties
    try:
        props = backend.properties(datetime=when)
    except Exception as e:
        print(f"Failed to fetch properties for {when}: {e}")
        return None

    # Build serializable object
    snapshot_iso = when.astimezone(dt.timezone.utc).isoformat()
    data = _props_to_serializable_dict(props, backend.name, snapshot_iso)

    # Ensure directory
    os.makedirs(out_dir, exist_ok=True)

    # Write file
    fname = f"{backend.name}_{_iso_filename(when.astimezone(dt.timezone.utc))}.json"
    fpath = os.path.join(out_dir, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return fpath


def _midnights_last_30(now_utc: dt.datetime) -> List[dt.datetime]:
    start_day = now_utc.date()
    marks: List[dt.datetime] = []
    for i in range(30):
        day = start_day - dt.timedelta(days=i)
        marks.append(dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc))
    return sorted(marks)


def main() -> None:
    out_dir = "backend_properties"

    mgr = QuantumServiceManager()
    if not mgr.connect():
        print("Could not connect to IBM Quantum. Check config/quantum_config.json")
        sys.exit(2)

    backend = mgr.select_backend()
    if backend is None:
        print("Failed to select backend from config.")
        sys.exit(1)

    now_utc = dt.datetime.now(dt.timezone.utc)
    marks = _midnights_last_30(now_utc)
    for when in marks:
        path = _save_snapshot_json(backend, when, out_dir)
        if path:
            print(f"Saved: {path}")


if __name__ == "__main__":
    main()