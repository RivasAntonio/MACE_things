#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extrae:
  - Energía: pymatgen (e_0_energy o e_fr_energy) desde vasprun.xml
  - Fuerzas + stress (+ virial derivado): ASE desde el MISMO vasprun.xml

y escribe un extxyz compatible con MACE (REF_energy, REF_forces, REF_stress, REF_virial).

Garantiza que pymatgen y ASE están en el mismo frame:
  - alinea por índice de ionic step
  - verifica que celda y posiciones coinciden (tolerancias configurables)
  - si no coincide, descarta el frame (o puedes cambiar el comportamiento)

Uso:
  python collect_mix.py --xml vasprun.xml --out frames.xyz --every 1 --verbose
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from ase.io import read as ase_read, write as ase_write
from pymatgen.io.vasp.outputs import Vasprun

# 1 eV/Å^3 = 1602.1766208 kbar  =>  1 kbar = 1/1602.1766208 eV/Å^3
KBAR_TO_EV_A3 = 1.0 / 1602.1766208

def pick_energy(step: dict, prefer=("e_0_energy", "e_fr_energy")) -> Tuple[Optional[float], Optional[str]]:
    for k in prefer:
        if k in step and step[k] is not None:
            return float(step[k]), k
    return None, None


def ase_pmg_same_frame(
    ase_atoms,
    pmg_structure,
    rtol_cell: float,
    atol_cell: float,
    rtol_pos: float,
    atol_pos: float,
) -> bool:
    """Comprueba si ASE y pymatgen representan el mismo frame (celda + coords cartesianas)."""
    cell_ase = np.asarray(ase_atoms.cell.array, dtype=float)
    cell_pmg = np.asarray(pmg_structure.lattice.matrix, dtype=float)
    if not np.allclose(cell_ase, cell_pmg, rtol=rtol_cell, atol=atol_cell):
        return False

    pos_ase = np.asarray(ase_atoms.get_positions(), dtype=float)
    pos_pmg = np.asarray(pmg_structure.cart_coords, dtype=float)

    # Misma cantidad de átomos
    if pos_ase.shape != pos_pmg.shape:
        return False

    # Ojo: si hubiese reordenación de especies, esto fallaría; en vasprun normal no debería ocurrir.
    if not np.allclose(pos_ase, pos_pmg, rtol=rtol_pos, atol=atol_pos):
        return False

    # Y opcionalmente: especies
    sym_ase = list(ase_atoms.get_chemical_symbols())
    sym_pmg = [str(sp) for sp in pmg_structure.species]
    if sym_ase != sym_pmg:
        return False

    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="vasprun.xml", help="vasprun.xml")
    ap.add_argument("--out", default="frames.xyz", help="salida extxyz")
    ap.add_argument("--every", type=int, default=1, help="submuestreo (cada N pasos)")
    ap.add_argument("--append", action="store_true", help="append si existe --out")
    ap.add_argument("--prefer_energy", default="e_0_energy,e_fr_energy",
                    help="orden de preferencia, separado por comas (default: e_0_energy,e_fr_energy)")
    ap.add_argument("--rtol_cell", type=float, default=1e-8)
    ap.add_argument("--atol_cell", type=float, default=1e-6)
    ap.add_argument("--rtol_pos", type=float, default=1e-8)
    ap.add_argument("--atol_pos", type=float, default=1e-5)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    xml = Path(args.xml)
    if not xml.exists():
        raise FileNotFoundError(str(xml))

    prefer = tuple([s.strip() for s in args.prefer_energy.split(",") if s.strip()])

    # ---- Pymatgen: energías (ionic_steps) ----
    vr = Vasprun(
        str(xml),
        parse_dos=False,
        parse_eigen=False,
        parse_projected_eigen=False,
        parse_potcar_file=False,
        exception_on_bad_xml=False,
    )
    pmg_steps = vr.ionic_steps
    n_pmg = len(pmg_steps)

    # ---- ASE: estructuras + fuerzas + stress desde el mismo vasprun.xml ----
    ase_frames = ase_read(str(xml), index=":", format="vasp-xml")
    n_ase = len(ase_frames)

    n = min(n_pmg, n_ase)
    if args.verbose:
        print(f"[INFO] steps: pymatgen={n_pmg}  ase={n_ase}  using min={n}")

    kept = 0
    skip_every = 0
    skip_missing = 0
    skip_energy = 0
    skip_mismatch = 0
    skip_stress = 0
    skip_forces = 0
    skip_nonfinite = 0

    out_frames = []

    incar = vr.incar or {}
    # Condicional solicitado: solo “aplicar corrección” si ISIF=3 e IBRION=0
    isif = int(incar.get("ISIF", 0)) if "ISIF" in incar else 0
    ibrion = int(incar.get("IBRION", 0)) if "IBRION" in incar else 0
    apply_pv = (isif == 3 and ibrion == 0)

    # PSTRESS: VASP lo interpreta en kB (numéricamente kbar)
    pstress_kbar = float(incar.get("PSTRESS", 0.0)) if "PSTRESS" in incar else 0.0

    if args.verbose:
        print(f"[INFO] ISIF={isif}  IBRION={ibrion}  PSTRESS={pstress_kbar} kbar  apply_-pV={apply_pv}")


    for i in range(n):
        if args.every > 1 and (i % args.every) != 0:
            skip_every += 1
            continue

        step = pmg_steps[i]
        ase_atoms = ase_frames[i]

        pmg_structure = step.get("structure", None)
        if pmg_structure is None:
            skip_missing += 1
            continue

        E, Ekey = pick_energy(step, prefer=prefer)
        if E is None:
            skip_energy += 1
            continue

        # Verificar alineamiento frame a frame
        if not ase_pmg_same_frame(
            ase_atoms,
            pmg_structure,
            rtol_cell=args.rtol_cell,
            atol_cell=args.atol_cell,
            rtol_pos=args.rtol_pos,
            atol_pos=args.atol_pos,
        ):
            skip_mismatch += 1
            continue

        # Fuerzas + stress con ASE (como en tu pipeline original)
        try:
            F = np.asarray(ase_atoms.get_forces(apply_constraint=False), dtype=float)
        except Exception:
            skip_forces += 1
            continue

        try:
            S = np.asarray(ase_atoms.get_stress(voigt=False), dtype=float)  # 3x3 en eV/Å^3
            if S.shape != (3, 3):
                skip_stress += 1
                continue
        except Exception:
            skip_stress += 1
            continue

        # Chequeos numéricos
        if (not np.isfinite(E)) or (not np.all(np.isfinite(F))) or (not np.all(np.isfinite(S))):
            skip_nonfinite += 1
            continue

        # Guardar en un Atoms “limpio” (sin calc), preservando cell/pos/especies
        ase_atoms.info["REF_energy_key_used"] = Ekey
        ase_atoms.arrays["REF_forces"] = F
        ase_atoms.info["REF_stress"] = S.reshape(9)

        V = float(ase_atoms.get_volume())
        pV_pstress_eV = pstress_kbar * V * KBAR_TO_EV_A3
        virial = -S * V  # eV
        
        ase_atoms.info["REF_virial"] = virial.reshape(9)
        ase_atoms.info["V_A3"] = V
        ase_atoms.info["ionic_step"] = i + 1
        ase_atoms.info["source"] = str(xml)

        if apply_pv:
            ase_atoms.info["REF_energy"] = float(E - pV_pstress_eV)
            ase_atoms.info["PSTRESS_kbar"] = float(pstress_kbar)
            ase_atoms.info["PSTRESS_pV_eV"] = float(pV_pstress_eV)
        else:
            ase_atoms.info["REF_energy"] = float(E)
	
        ase_atoms.calc = None
        out_frames.append(ase_atoms)
        kept += 1

    if args.verbose:
        print("Resumen filtrado:")
        print(f"  total(min): {n}")
        print(f"  kept: {kept}")
        print(f"  skip_every: {skip_every}")
        print(f"  skip_missing_structure: {skip_missing}")
        print(f"  skip_no_energy: {skip_energy}")
        print(f"  skip_mismatch_ase_pmg: {skip_mismatch}")
        print(f"  skip_bad_forces: {skip_forces}")
        print(f"  skip_bad_stress: {skip_stress}")
        print(f"  skip_nonfinite: {skip_nonfinite}")

    mode_append = bool(args.append and Path(args.out).exists())
    ase_write(args.out, out_frames, format="extxyz", append=mode_append)
    print(f"OK: escritos {kept} frames en {args.out}")


if __name__ == "__main__":
    main()

