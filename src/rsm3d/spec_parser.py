#!/usr/bin/env python3
"""
spec_parser.py

Defines ExperimentSetup, Crystal, ScanAngles, and SpecParser classes to parse
SPEC-format files and return scan data as a pandas or Dask DataFrame.
"""

import sys
import numpy as np
import pandas as pd
import dask.dataframe as dd
import yaml
from pathlib import Path


class ExperimentSetup:
    """
    Load experiment parameters from a YAML file. Wavelength is optional:
      • if provided and >1e-3 Å, used directly
      • if provided in meters (<1e-3), converted to Å
      • if omitted or non‐positive, computed from energy [Å] = 12.398419843320026 / E[keV]

    Required keys (either top-level or inside `ExperimentSetup:`):
      distance, pitch, ycenter, xcenter, xpixels, ypixels, energy

    Optional key:
      wavelength
    """
    REQUIRED_KEYS = (
        "distance", "pitch", "ycenter", "xcenter",
        "xpixels", "ypixels", "energy",
    )

    def __init__(
        self,
        distance: float,
        pitch: float,
        ycenter: int,
        xcenter: int,
        xpixels: int,
        ypixels: int,
        energy: float,
        wavelength: float | None = None,
    ):
        self.distance = float(distance)
        self.pitch = float(pitch)
        self.ycenter = int(ycenter)
        self.xcenter = int(xcenter)
        self.xpixels = int(xpixels)
        self.ypixels = int(ypixels)
        self.energy = float(energy)
        self.energy_keV = float(energy)
        if self.distance <= 0:
            raise ValueError("ExperimentSetup: 'distance' must be > 0")
        if self.pitch <= 0:
            raise ValueError("ExperimentSetup: 'pitch' must be > 0")
        if self.xpixels <= 0 or self.ypixels <= 0:
            raise ValueError("ExperimentSetup: 'xpixels' and 'ypixels' must be > 0")
        if self.energy_keV <= 0:
            raise ValueError("ExperimentSetup: 'energy' (keV) must be > 0")

        lam_A: float | None = None
        if wavelength is not None:
            try:
                lam_A = float(wavelength)
            except (TypeError, ValueError):
                lam_A = None
        if lam_A is not None and 0.0 < lam_A < 1e-3:
            lam_A *= 1e10
        if lam_A is None or lam_A <= 0.0:
            lam_A = self._energy_keV_to_lambda_A(self.energy_keV)
        if lam_A <= 0.0:
            raise ValueError("ExperimentSetup: computed wavelength is non-positive")
        self.wavelength = lam_A

    @staticmethod
    def _energy_keV_to_lambda_A(E_keV: float) -> float:
        return 12.398419843320026 / float(E_keV)

    @staticmethod
    def _to_float(v):
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            try:
                return float(str(v).replace("_", "").strip())
            except Exception:
                raise ValueError(f"Expected float-compatible value, got {v!r}")

    @staticmethod
    def _to_int(v):
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            try:
                return int(float(str(v).replace("_", "").strip()))
            except Exception:
                raise ValueError(f"Expected int-compatible value, got {v!r}")

    @classmethod
    def _extract_section(cls, data: dict) -> dict:
        if not isinstance(data, dict):
            raise ValueError("Top-level YAML must be a mapping of keys to values.")
        for key in ("ExperimentSetup", "experiment", "experiment_setup"):
            sec = data.get(key)
            if isinstance(sec, dict):
                return sec
        if any(k in data for k in cls.REQUIRED_KEYS):
            return data
        for v in data.values():
            if isinstance(v, dict) and any(k in v for k in cls.REQUIRED_KEYS):
                return v
        raise ValueError(
            "Could not find experiment setup in YAML. "
            "Expected an 'ExperimentSetup' section or flat keys."
        )

    @classmethod
    def from_yaml(cls, path: str | Path):
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Experiment YAML not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        sec = cls._extract_section(doc)
        merged = {}
        for k in cls.REQUIRED_KEYS + ("wavelength",):
            if k in sec:
                merged[k] = sec[k]
            elif k in doc:
                merged[k] = doc[k]
        missing = [k for k in cls.REQUIRED_KEYS if merged.get(k) in (None, "", "None", "null")]
        if missing:
            raise ValueError(f"Missing required keys in YAML: {missing}")
        params = {
            "distance":  cls._to_float(merged["distance"]),
            "pitch":     cls._to_float(merged["pitch"]),
            "ycenter":   cls._to_int(merged["ycenter"]),
            "xcenter":   cls._to_int(merged["xcenter"]),
            "xpixels":   cls._to_int(merged["xpixels"]),
            "ypixels":   cls._to_int(merged["ypixels"]),
            "energy":    cls._to_float(merged["energy"]),
            "wavelength": merged.get("wavelength", None),
        }
        return cls(**params)

    def __repr__(self):
        return (
            f"<ExperimentSetup: distance={self.distance} m, pitch={self.pitch} m, "
            f"xcenter={self.xcenter}, ycenter={self.ycenter}, "
            f"xpixels={self.xpixels}, ypixels={self.ypixels}, "
            f"energy={self.energy} keV, wavelength={self.wavelength} Å>"
        )


class Crystal:
    """Holds crystal lattice parameters and orientation matrix (UB)."""
    def __init__(self, params, ub_matrix):
        self.a, self.b, self.c = params[0:3]
        self.alpha, self.beta, self.gamma = params[3:6]
        self.a_hkl, self.b_hkl, self.c_hkl = params[6:9]
        self.alpha_hkl, self.beta_hkl, self.gamma_hkl = params[9:12]
        self.H_or0, self.K_or0, self.L_or0 = params[12:15]
        self.H_or1, self.K_or1, self.L_or1 = params[15:18]
        (self.u00, self.u01, self.u02,
         self.u03, self.u04, self.u05) = params[18:24]
        (self.u10, self.u11, self.u12,
         self.u13, self.u14, self.u15) = params[24:30]
        self.lambda0, self.lambda1 = params[30:32]
        self.u06, self.u16 = params[32:34]
        ub = np.array(ub_matrix, dtype=float)
        self.UB = ub.reshape((3, 3))

    @classmethod
    def from_spec(cls, filename):
        g1_vals, g3_vals = None, None
        with open(filename) as f:
            for line in f:
                if line.startswith('#G1 '):
                    g1_vals = [float(x) for x in line.split()[1:]]
                elif line.startswith('#G3 '):
                    g3_vals = [float(x) for x in line.split()[1:]]
                if g1_vals is not None and g3_vals is not None:
                    break
        if g1_vals is None or g3_vals is None:
            raise RuntimeError("Missing #G1 or #G3 in SPEC file for Crystal.")
        return cls(g1_vals, g3_vals)

    def __repr__(self):
        return (
            f"Crystal(a={self.a}, b={self.b}, c={self.c}, alpha={self.alpha}, "
            f"beta={self.beta}, gamma={self.gamma}, UB=\n{self.UB})"
        )


class ScanAngles:
    ASCAN_AXES = ('VTTH', 'VTH', 'Phi', 'Chi')
    HKL_AXES   = ('VTTH', 'VTH', 'Chi', 'Phi')

    def __init__(self, filename, crystal, npartitions=1):
        self.filename = filename
        self.npartitions = npartitions
        self.crystal = crystal

    def parse_all_scans(self):
        o0_names = []
        with open(self.filename) as f:
            for line in f:
                if line.startswith('#O0 '):
                    o0_names = line[4:].split()
                    break
        if not o0_names:
            raise RuntimeError("Missing global #O0 line in SPEC file")

        results = []
        cur_scan = None
        cur_type = None
        p0_map = {}
        data_idx = {}
        in_data = False
        counter = 0
        current_ub = None
        skip_scan = False
        scan_results_start = 0  # index to roll back if skipping

        with open(self.filename) as f:
            for raw in f:
                line = raw.strip()

                if line.startswith('#S '):
                    # new scan
                    parts = line.split()
                    cur_scan = int(parts[1])
                    cur_type = parts[2] if len(parts) > 2 else ''
                    p0_map.clear()
                    data_idx.clear()
                    in_data = False
                    counter = 0
                    current_ub = None
                    skip_scan = False
                    scan_results_start = len(results)
                    continue

                if skip_scan:
                    # keep consuming until next #S
                    continue

                if cur_scan is not None and line.startswith('#G3 '):
                    ub_vals = [float(x) for x in line.split()[1:]]
                    if len(ub_vals) == 9:
                        current_ub = np.array(ub_vals).reshape((3, 3))
                    continue

                if cur_scan is not None and line.startswith('#P0 '):
                    vals = [float(x) for x in line.split()[1:]]
                    p0_map = {name: vals[i] for i, name in enumerate(o0_names)}
                    continue

                if cur_scan is not None and line.startswith('#L '):
                    cols = line.split()[1:]
                    # Decide if we can obtain all four angles
                    required_angles = {'VTTH', 'VTH', 'Chi', 'Phi'}

                    if cur_type.lower() == 'ascan':
                        axes = [c for c in self.ASCAN_AXES if c in cols]
                        if len(axes) != 1:
                            # cannot determine which axis was scanned or ambiguous
                            print(f"[spec_parser] Ignoring scan {cur_scan}: ambiguous or missing scan axis.")
                            skip_scan = True
                            continue
                        scan_col = axes[0]
                        data_idx['scan_col'] = cols.index(scan_col)
                        # H,K,L must exist
                        need_cols = {'H', 'K', 'L'}
                        if not need_cols.issubset(cols):
                            print(f"[spec_parser] Ignoring scan {cur_scan}: missing H/K/L columns.")
                            skip_scan = True
                            continue
                        for hk in ('H', 'K', 'L'):
                            data_idx[hk] = cols.index(hk)
                        # For ascan, other three fixed angles must be in p0_map
                        fixed_needed = required_angles - {scan_col}
                        if any(p0_map.get(a) is None for a in fixed_needed):
                            print(f"[spec_parser] Ignoring scan {cur_scan}: missing fixed angle values {fixed_needed}.")
                            skip_scan = True
                            continue

                    elif cur_type.lower() == 'hklscan':
                        # All four angle axes must be present in columns
                        if not required_angles.issubset(cols):
                            missing = required_angles - set(cols)
                            print(f"[spec_parser] Ignoring scan {cur_scan}: missing angle columns {missing}.")
                            skip_scan = True
                            continue
                        # Index angles
                        try:
                            for ax in self.HKL_AXES:
                                data_idx[ax] = cols.index(ax)
                        except ValueError as e:
                            print(f"[spec_parser] Ignoring scan {cur_scan}: angle index error ({e}).")
                            skip_scan = True
                            continue
                        # H,K,L must exist
                        need_cols = {'H', 'K', 'L'}
                        if not need_cols.issubset(cols):
                            print(f"[spec_parser] Ignoring scan {cur_scan}: missing H/K/L columns.")
                            skip_scan = True
                            continue
                        for hk in ('H', 'K', 'L'):
                            data_idx[hk] = cols.index(hk)
                    else:
                        # unsupported scan type — ignore
                        print(f"[spec_parser] Ignoring scan {cur_scan}: unsupported scan type '{cur_type}'.")
                        skip_scan = True
                        continue

                    # Ready to read data rows for this scan
                    in_data = True
                    continue

                if skip_scan:
                    continue

                if in_data:
                    if not line or (line.startswith('#') and not line[1].isdigit()):
                        in_data = False
                        continue
                    parts = line.split()
                    if len(parts) < max(data_idx.values()) + 1:
                        continue

                    rec = {
                        'scan_number': f"{cur_scan:03d}",
                        'data_number': f"{counter:03d}",
                        'type': cur_type,
                        'ub': current_ub.copy() if current_ub is not None else None
                    }

                    if cur_type.lower() == 'ascan':
                        rec.update({
                            'tth': p0_map.get('VTTH'),
                            'th':  p0_map.get('VTH'),
                            'chi': p0_map.get('Chi'),
                            'phi': p0_map.get('Phi'),
                            'h':   float(parts[data_idx['H']]),
                            'k':   float(parts[data_idx['K']]),
                            'l':   float(parts[data_idx['L']])
                        })
                        val = float(parts[data_idx['scan_col']])
                        scan_col = self.ASCAN_AXES[[c for c in self.ASCAN_AXES if c in data_idx].index('scan_col')] if 'scan_col' in data_idx else None
                        # direct mapping with earlier stored scan_col variable (safer):
                        # we retained 'scan_col' variable in the closure scope
                        if 'scan_col' in data_idx:
                            sc_idx = data_idx['scan_col']
                            scanned_value = float(parts[sc_idx])
                            if scan_col == 'VTTH':
                                rec['tth'] = scanned_value
                            elif scan_col == 'VTH':
                                rec['th'] = scanned_value
                            elif scan_col == 'Phi':
                                rec['phi'] = scanned_value
                            elif scan_col == 'Chi':
                                rec['chi'] = scanned_value
                    else:
                        rec.update({
                            'tth': float(parts[data_idx['VTTH']]),
                            'th':  float(parts[data_idx['VTH']]),
                            'chi': float(parts[data_idx['Chi']]),
                            'phi': float(parts[data_idx['Phi']]),
                            'h':   float(parts[data_idx['H']]),
                            'k':   float(parts[data_idx['K']]),
                            'l':   float(parts[data_idx['L']])
                        })

                    # Validate presence of all four angles; if any missing -> skip entire scan
                    if any(rec[a] is None for a in ('tth', 'th', 'chi', 'phi')):
                        print(f"[spec_parser] Ignoring scan {cur_scan}: incomplete angle data in rows.")
                        # remove any partial rows for this scan
                        results = results[:scan_results_start]
                        skip_scan = True
                        in_data = False
                        continue

                    results.append(rec)
                    counter += 1

        return results

    def to_pandas(self):
        return pd.DataFrame(self.parse_all_scans())

    def to_dask(self):
        return dd.from_pandas(self.to_pandas(), npartitions=self.npartitions)


class SpecParser:
    """Aggregates ExperimentSetup, Crystal, and ScanAngles for a SPEC file."""
    def __init__(self, filename: str, setup_yaml: str, npartitions: int = 1):
        self.filename = filename
        self.setup = ExperimentSetup.from_yaml(setup_yaml)
        self.crystal = Crystal.from_spec(filename)
        self.scans = ScanAngles(filename, self.crystal, npartitions=npartitions)

    def to_pandas(self):
        return self.scans.to_pandas()

    def to_dask(self):
        return self.scans.to_dask()