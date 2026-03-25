#!/usr/bin/env python3
"""
Creates a supercell of FA,MA Pb Br,I3 perovskite that can include vacancies.
The distribution of the different atoms and molecules is extracted from SQS algorithm.
Generates output in POSCAR format.

Vacancy charge-neutrality rules (Schottky defects):
  - 1 A-site vacancy (V_A, charge -1) must be compensated by 1 X-site vacancy (V_X, charge +1).
  - 1 B-site vacancy (V_B, charge -2) must be compensated by 2 X-site vacancies.
  - Combined: charge_balance = n_FA + n_MA + 2*n_Pb - n_I - n_Br == 0
  These can be mixed freely as long as the net formal charge is zero.
"""

import argparse
import numpy as np
import sys
from ase import Atoms
from ase.build import make_supercell
from icet import ClusterSpace
from icet.tools.structure_generation import generate_sqs_from_supercells
from icet.input_output.logging_tools import set_log_config


# ---------------------------------------------------------------------------
# Input / parameter helpers
# ---------------------------------------------------------------------------

def leer_input(filepath="input_supercell_creator"):
    """Lee y extrae los 11 parámetros obligatorios del archivo de entrada."""
    try:
        with open(filepath, 'r') as f:
            lines = [line.split() for line in f if line.strip()]
        return list(map(int, [line[0] for line in lines[:11]]))
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        sys.exit(1)


def calcular_parametro_red(n_FA, n_MA, n_I, n_Br):
    """Aplica la Ley de Vegard para estimar la constante de red cúbica."""
    tot_A = max(1, n_FA + n_MA)
    tot_X = max(1, n_I + n_Br)

    f_FA, f_MA = n_FA / tot_A, n_MA / tot_A
    f_I,  f_Br = n_I  / tot_X, n_Br / tot_X

    # Constantes de red de fases puras teóricas (Å)
    a = (f_FA * f_I * 6.35 + f_FA * f_Br * 5.99 +
         f_MA * f_I * 6.30 + f_MA * f_Br * 5.92)
    return a if a > 0 else 6.30


# ---------------------------------------------------------------------------
# Charge-neutrality validator
# ---------------------------------------------------------------------------

def verificar_neutralidad(n_FA, n_MA, n_vac_A,
                          n_Pb, n_vac_B,
                          n_I,  n_Br,  n_vac_X,
                          allow_charged=False):
    """
    Checks formal charge neutrality including vacancies.

    Formal charges: FA⁺(+1), MA⁺(+1), Pb²⁺(+2), I⁻(-1), Br⁻(-1).
    A vacancy on a site removes the ion's charge from the lattice, which is
    equivalent to introducing the opposite formal charge:
        V_A  → effectively −(+1) = −1
        V_B  → effectively −(+2) = −2
        V_X  → effectively −(−1) = +1

    Therefore the net charge is:
        Q = n_FA(+1) + n_MA(+1) + n_Pb(+2) + n_I(−1) + n_Br(−1)
          = n_FA + n_MA + 2·n_Pb − n_I − n_Br

    Valid charge-neutral vacancy combinations (Schottky pairs/triples):
        1 V_A  +  1 V_X  →  (−1) + (+1) = 0   ✓
        1 V_B  +  2 V_X  →  (−2) + 2·(+1) = 0 ✓
        Any linear combination of the above                          ✓

    Parameters
    ----------
    allow_charged : bool
        If True, a non-zero charge only raises a warning instead of aborting.
    """
    carga = n_FA + n_MA + 2 * n_Pb - n_I - n_Br

    if carga == 0:
        # Report vacancy types present for transparency
        vac_lines = []
        if n_vac_A > 0:
            vac_lines.append(f"  V_A (A-site): {n_vac_A}")
        if n_vac_B > 0:
            vac_lines.append(f"  V_B (B-site): {n_vac_B}")
        if n_vac_X > 0:
            vac_lines.append(f"  V_X (X-site): {n_vac_X}")
        if vac_lines:
            print("Vacancies detected (charge-neutral configuration):")
            print("\n".join(vac_lines))
            # Verify Schottky balance explicitly
            schottky_check = n_vac_A + 2 * n_vac_B - n_vac_X
            if schottky_check != 0:
                print(
                    "  Warning: vacancy counts do not form integer Schottky pairs "
                    f"(V_A + 2·V_B − V_X = {schottky_check} ≠ 0), but overall "
                    "charge happens to be neutral due to ion stoichiometry."
                )
        return  # OK

    # Non-neutral system
    msg_lines = [
        f"",
        f"  Charge error: net formal charge Q = {carga:+d}",
        f"  Contributions: n_FA={n_FA}(+1), n_MA={n_MA}(+1), n_Pb={n_Pb}(+2), "
        f"n_I={n_I}(-1), n_Br={n_Br}(-1)",
        f"",
        f"  To neutralise the system with vacancies use Schottky combinations:",
        f"    • 1 A-site vacancy (V_A) + 1 X-site vacancy (V_X)  →  ΔQ = 0",
        f"    • 1 B-site vacancy (V_B) + 2 X-site vacancies (V_X) →  ΔQ = 0",
        f"",
        f"  With the current vacancy counts (V_A={n_vac_A}, V_B={n_vac_B}, V_X={n_vac_X}):",
        f"    Required V_X to compensate = n_vac_A + 2·n_vac_B = {n_vac_A + 2*n_vac_B}",
        f"    Provided V_X                                      = {n_vac_X}",
        f"    Deficit / surplus of V_X                         = {n_vac_X - (n_vac_A + 2*n_vac_B):+d}",
    ]

    if allow_charged:
        print("WARNING (--allow-charged flag active): system is NOT charge-neutral.")
        print("\n".join(msg_lines))
        print()
    else:
        print("ERROR: system is NOT charge-neutral.")
        print("\n".join(msg_lines))
        print()
        print("  Use --allow-charged to override this check (e.g. for charged-defect calculations).")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Molecular geometry data
# ---------------------------------------------------------------------------

# Símbolos y coordenadas fraccionales de FA (Formamidinio) centradas en (0.5, 0.5, 0.5)
FA_SYMBOLS = ['C', 'N', 'N', 'H', 'H', 'H', 'H', 'H']
coord_frac_FA = np.array([
    [0.50000, 0.56901, 0.5],   # C
    [0.68293, 0.47528, 0.5],   # N
    [0.31707, 0.47528, 0.5],   # N
    [0.50000, 0.74142, 0.5],   # Hc
    [0.81400, 0.56763, 0.5],   # H
    [0.70447, 0.31528, 0.5],   # H
    [0.29554, 0.31528, 0.5],   # H
    [0.18599, 0.56763, 0.5]    # H
])

# Símbolos y coordenadas fraccionales de MA (Metilamonio) centradas en (0.5, 0.5, 0.5)
MA_SYMBOLS = ['C', 'N', 'H', 'H', 'H', 'H', 'H', 'H']
coord_frac_MA = np.array([
    [0.42600, 0.50000, 0.48410],  # C
    [0.65411, 0.50000, 0.54685],  # N
    [0.41270, 0.50000, 0.30200],  # Hc
    [0.35250, 0.65241, 0.55342],  # Hc
    [0.35230, 0.34790, 0.55360],  # Hc
    [0.73222, 0.64233, 0.48690],  # H
    [0.73209, 0.35730, 0.48690],  # H
    [0.67712, 0.50000, 0.71846]   # H
])

# The fractional origin that the above coordinates are measured from
CENTRO_SITIO_A = np.array([0.5, 0.5, 0.5])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog='perovskite_poscar_creator_sqs.py',
        description=(
            'Genera una supercelda POSCAR de perovskita mixta FA/MA Pb (Br/I)3 '
            'con distribución cuasialeatoria (SQS) mediante el algoritmo de icet.\n\n'
            'Vacancies are supported as charge-neutral Schottky defects:\n'
            '  1 V_A + 1 V_X  or  1 V_B + 2 V_X  (any integer combination).\n'
            'Use --allow-charged to bypass the neutrality check.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '-i', '--input',
        default='input_supercell_creator',
        metavar='ARCHIVO',
        help='Ruta al archivo de entrada (default: input_supercell_creator)'
    )
    parser.add_argument(
        '--allow-charged',
        action='store_true',
        help=(
            'Allow non-neutral charge (e.g. for charged-defect DFT calculations '
            'with a compensating background charge). Emits a warning instead of aborting.'
        )
    )
    parser.add_argument(
        '-o', '--output',
        default='POSCAR',
        metavar='ARCHIVO_SALIDA',
        help='Nombre del archivo POSCAR de salida (default: POSCAR)'
    )
    args = parser.parse_args()

    set_log_config(level='INFO')

    # ------------------------------------------------------------------
    # 1. Read and validate site occupancy counts
    # ------------------------------------------------------------------
    nx, ny, nz, n_FA, n_MA, n_vac_A, n_Pb, n_vac_B, n_I, n_Br, n_vac_X = leer_input(args.input)
    N_celdas = nx * ny * nz

    if (n_FA + n_MA + n_vac_A) != N_celdas:
        print(
            f"Error: A-site occupants ({n_FA}+{n_MA}+{n_vac_A}={n_FA+n_MA+n_vac_A}) "
            f"do not match supercell size ({N_celdas})."
        )
        sys.exit(1)
    if (n_Pb + n_vac_B) != N_celdas:
        print(
            f"Error: B-site occupants ({n_Pb}+{n_vac_B}={n_Pb+n_vac_B}) "
            f"do not match supercell size ({N_celdas})."
        )
        sys.exit(1)
    if (n_I + n_Br + n_vac_X) != 3 * N_celdas:
        print(
            f"Error: X-site occupants ({n_I}+{n_Br}+{n_vac_X}={n_I+n_Br+n_vac_X}) "
            f"do not match 3 × supercell size ({3*N_celdas})."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Charge-neutrality check (vacancies accounted for)
    # ------------------------------------------------------------------
    verificar_neutralidad(
        n_FA, n_MA, n_vac_A,
        n_Pb, n_vac_B,
        n_I,  n_Br,  n_vac_X,
        allow_charged=args.allow_charged
    )

    # ------------------------------------------------------------------
    # 3. Build icet ClusterSpace with pseudo-elements
    #    Vacancy markers: He → V_A,  Ne → V_B,  Ar → V_X
    # ------------------------------------------------------------------
    a = calcular_parametro_red(n_FA, n_MA, n_I, n_Br)

    # Primitive cell: K placeholder for the A-site
    prim = Atoms(
        'KPbI3',
        scaled_positions=[
            (0.5, 0.5, 0.5),   # A-site  (K placeholder)
            (0.0, 0.0, 0.0),   # B-site  (Pb)
            (0.5, 0.0, 0.0),   # X-site  (I)
            (0.0, 0.5, 0.0),   # X-site  (I)
            (0.0, 0.0, 0.5),   # X-site  (I)
        ],
        cell=[a, a, a],
        pbc=True,
    )

    # Build species lists — include a species only when its count is > 0
    A_specs = [sp for sp, c in [('Fr', n_FA), ('Cs', n_MA), ('He', n_vac_A)] if c > 0]
    B_specs = [sp for sp, c in [('Pb', n_Pb), ('Ne', n_vac_B)]               if c > 0]
    X_specs = [sp for sp, c in [('I',  n_I),  ('Br', n_Br),  ('Ar', n_vac_X)] if c > 0]

    chem_symbols = [A_specs, B_specs, X_specs, X_specs, X_specs]

    # ------------------------------------------------------------------
    # Short-circuit for pure (single-species) compositions:
    # icet's ClusterSpace requires at least one sublattice with >1 species
    # ("active site").  When x=0/1 or y=0/1 the structure is fully ordered
    # and SQS is unnecessary — build the ordered supercell directly instead.
    # ------------------------------------------------------------------
    is_pure = all(len(s) == 1 for s in [A_specs, B_specs, X_specs])
    supercell_matrix = np.diag([nx, ny, nz])

    if is_pure:
        print(
            f"\nPure (single-species) composition — skipping SQS, "
            f"building ordered {nx}×{ny}×{nz} supercell directly …"
        )
        # Replace prim placeholders (K, Pb, I) with the actual pseudo-element
        placeholder_map = {'K': A_specs[0], 'Pb': B_specs[0], 'I': X_specs[0]}
        sqs_structure = make_supercell(prim, supercell_matrix)
        sqs_structure.set_chemical_symbols(
            [placeholder_map[s] for s in sqs_structure.get_chemical_symbols()]
        )
    else:
        cs = ClusterSpace(structure=prim, cutoffs=[6.0], chemical_symbols=chem_symbols)

        # Target concentrations for each sublattice
        target_concs = {}
        for sl in cs.get_sublattices(prim):
            syms = set(sl.chemical_symbols)
            if syms & {'Fr', 'Cs', 'He'}:
                target_concs[sl.symbol] = {
                    k: v for k, v in {
                        'Fr': n_FA    / N_celdas,
                        'Cs': n_MA    / N_celdas,
                        'He': n_vac_A / N_celdas,
                    }.items() if v > 0
                }
            elif syms & {'Pb', 'Ne'}:
                target_concs[sl.symbol] = {
                    k: v for k, v in {
                        'Pb': n_Pb    / N_celdas,
                        'Ne': n_vac_B / N_celdas,
                    }.items() if v > 0
                }
            elif syms & {'I', 'Br', 'Ar'}:
                target_concs[sl.symbol] = {
                    k: v for k, v in {
                        'I':  n_I    / (3 * N_celdas),
                        'Br': n_Br   / (3 * N_celdas),
                        'Ar': n_vac_X / (3 * N_celdas),
                    }.items() if v > 0
                }

        # ------------------------------------------------------------------
        # 4. SQS optimisation
        # ------------------------------------------------------------------
        empty_supercell = make_supercell(prim, supercell_matrix)

        print(
            f"\nGenerating SQS model via icet  "
            f"({nx}×{ny}×{nz} supercell, {N_celdas} formula units) …"
        )
        print(
            f"Vacancy summary: V_A={n_vac_A}, V_B={n_vac_B}, V_X={n_vac_X}"
            f"  (pseudo-elements He, Ne, Ar respectively)\n"
        )

        sqs_structure = generate_sqs_from_supercells(
            cluster_space=cs,
            supercells=[empty_supercell],
            target_concentrations=target_concs,
            n_steps=100_000,
            T_start=5.0,
            T_stop=0.001,
        )

    # ------------------------------------------------------------------
    # 5. Translate SQS pseudo-atoms → physical atoms
    #    • Fr  → expand FA molecule using primitive-cell fractional offsets
    #    • Cs  → expand MA molecule
    #    • Pb, I, Br → keep as-is
    #    • He (V_A), Ne (V_B), Ar (V_X) → silently dropped (vacancies)
    # ------------------------------------------------------------------
    final_symbols   = []
    final_positions = []

    # Pre-compute the 3×3 lattice matrix of the *primitive* cell so that
    # fractional offsets are correctly mapped to Cartesian space regardless
    # of whether the supercell is cubic or not.
    prim_cell = np.array(prim.cell)   # shape (3, 3), rows are lattice vectors

    for atom in sqs_structure:
        sym = atom.symbol
        pos = atom.position   # Cartesian position of this site in the supercell

        if sym == 'Fr':   # FA⁺
            for elem, frac_coord in zip(FA_SYMBOLS, coord_frac_FA):
                delta_frac = frac_coord - CENTRO_SITIO_A          # fractional offset
                delta_cart = delta_frac @ prim_cell                # → Cartesian (Å)
                final_symbols.append(elem)
                final_positions.append(pos + delta_cart)

        elif sym == 'Cs':  # MA⁺
            for elem, frac_coord in zip(MA_SYMBOLS, coord_frac_MA):
                delta_frac = frac_coord - CENTRO_SITIO_A
                delta_cart = delta_frac @ prim_cell
                final_symbols.append(elem)
                final_positions.append(pos + delta_cart)

        elif sym in ('Pb', 'I', 'Br'):
            final_symbols.append(sym)
            final_positions.append(pos)

        # He / Ne / Ar  →  vacancy: atom intentionally omitted

    # ------------------------------------------------------------------
    # 6. Build final Atoms object and sort species: C, N, H, Pb, I, Br
    # ------------------------------------------------------------------
    final_atoms = Atoms(
        symbols=final_symbols,
        positions=final_positions,
        cell=sqs_structure.cell,
        pbc=True,
    )

    orden = {'C': 0, 'N': 1, 'H': 2, 'Pb': 3, 'I': 4, 'Br': 5}
    indices_ordenados = sorted(
        range(len(final_atoms)),
        key=lambda i: orden.get(final_atoms[i].symbol, 99),
    )
    final_atoms = final_atoms[indices_ordenados]

    final_atoms.write(args.output, format="vasp")
    print(f"\n✓ Quasi-random structure saved to {args.output}")
    print(f"  Total atoms in cell: {len(final_atoms)}")

    # Composition summary
    from collections import Counter
    comp = Counter(final_atoms.get_chemical_symbols())
    comp_str = "  Composition: " + "  ".join(
        f"{sym}={comp[sym]}" for sym in ['C', 'N', 'H', 'Pb', 'I', 'Br'] if sym in comp
    )
    print(comp_str)

    if n_vac_A + n_vac_B + n_vac_X > 0:
        print(
            f"  Vacancies introduced: V_A={n_vac_A}, V_B={n_vac_B}, V_X={n_vac_X}  "
            f"(sites left empty in POSCAR)"
        )


if __name__ == "__main__":
    main()
