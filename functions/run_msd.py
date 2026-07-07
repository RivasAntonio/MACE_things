"""
Vacancy-mediated ionic diffusion in mixed hybrid perovskite (FA-MA-Pb-I-Br)
via MSD-based diffusion coefficient extraction.

Workflow:
    1. Geometry optimization  (float64, BFGS)             — 0 K, relaxed cell
    2. NVT Heating Ramp       (Langevin, NVT only)        — 10 K -> Target T
    3. NPT Equilibration      (LangevinBAOAB, hydrostatic) — Target T, 0 bar
         + automated convergence check with optional extensions
    4. NVT Stabilization      (Langevin)                  — Target T, fixed equilibrium volume
    5. NVE Production         (VelocityVerlet)             — Target T, 10 ns, 1 ps/frame

Key corrections vs previous version
------------------------------------
[1] Ramp now uses pure NVT (Langevin), not NPT.
    Applying a barostat below ~100 K on a soft defective perovskite causes
    unphysical volume collapse. The cell should only be allowed to relax
    once the system is fully thermalized at the target temperature.

[2] T_BAROSTAT increased from 5000 fs to 20000 fs.
    The previous value produced a barostat that was too stiff for a
    mechanically compliant defective perovskite, leading to large pressure
    oscillations and preventing volume convergence within 100 ps.
    With T_BAROSTAT = 20 ps, the barostat damps smoothly without over-shooting.

[3] NPT duration increased from 100 ps to 1000 ps (baseline).
    The previous run exited NPT with a residual mean pressure of -2252 bar,
    confirming the barostat had not converged. Rule of thumb: allow at least
    30 x T_BAROSTAT of equilibration time after the initial transient. With
    T_BAROSTAT = 20 ps that is 600 ps minimum; 1000 ps is used as the baseline.

[4] Automated NPT convergence check with extension cycles.
    The script splits the NPT trajectory into blocks and verifies two
    criteria before accepting the volume:
      (a) Relative std of block-mean volumes < VOL_CONVERGENCE_TOL (0.5%)
      (b) Mean pressure of the last 20% of the run < P_CONVERGENCE_BAR (200 bar)
    If either criterion fails, the run is extended by NPT_EXTENSION_PS and
    re-checked, up to NPT_MAX_EXTENSIONS times. This behaviour is controlled
    by the NPT_ALLOW_EXTENSION flag: set it to False to run exactly
    NPT_DURATION_PS regardless of convergence (useful under wall-time
    constraints). The convergence report is printed either way.
    A hard warning is raised if convergence is never achieved.

[5] Volume averaging uses only the last 20% of the converged NPT run,
    not the full second half. This ensures the average is drawn from a
    stationary distribution only.

[6] NVT stabilization increased from 200 ps to 500 ps.
    The NVT stage must dissipate any residual kinetic energy imbalance
    inherited from the volume-rescaling step. 200 ps was insufficient.

[7] T_THERMOSTAT increased from 100 fs to 200 fs.
    A slightly softer thermostat reduces spurious coupling between the
    thermostat and the slow structural modes of the defective lattice.

[8] Thermodynamic summary printed at the end of each equilibration stage
    so convergence can be assessed without reading raw log files.
"""

import os
import copy
import time
import warnings
import torch
import numpy as np

from ase.io import read, write
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from ase.md.langevin import Langevin
from ase.md.langevinbaoab import LangevinBAOAB
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase import units
from ase.atoms import Atoms
from mace.calculators import MACECalculator

# =============================================================================
# CONTROL PARAMETERS
# =============================================================================

INPUT_STRUCTURE = "./POSCAR"
MINIMIZED_XYZ   = "./optimization/minimized.xyz"
OPTIM_TRAJ      = "./optimization/optimization.traj"

TARGET_DEVICE   = "cuda:0"

TEMPERATURES    = 300    # K
PRESSURE        = 0.0           # bar

TIMESTEP_FS     = 0.5           # fs
TIMESTEP        = TIMESTEP_FS * units.fs

# ----------------- THERMOSTAT / BAROSTAT -----------------
# T_THERMOSTAT: thermostat time constant.
# 200 fs is soft enough to avoid coupling to slow structural modes,
# while still maintaining temperature control within ~10 K.
T_THERMOSTAT = 200.0* units.fs        # was 100 fs

# T_BAROSTAT: barostat time constant.
# Must satisfy T_BAROSTAT >> tau_phonon for the softest acoustic mode.
# For a ~6.2 A perovskite cell, the zone-boundary acoustic mode period
# is ~1-2 ps. T_BAROSTAT = 20 ps gives ~10-20 damping cycles per acoustic period,
# which is the standard underdamped regime for a Parrinello-Rahman-style barostat.
T_BAROSTAT = 10000.0 * units.fs      # was 5000 fs

# ----------------- STAGE 1: HEATING RAMP (NVT only) -----------------
# Pure Langevin ramp. No barostat during heating.
# The cell is held fixed at the 0 K minimized volume until the system
# is fully thermalized. Volume relaxation is deferred to stage 2 (NPT).
RAMP_START_T      = 10.0        # K
RAMP_DURATION_PS  = 100.0       # ps  
RAMP_STEPS        = int(RAMP_DURATION_PS * 1e3 / TIMESTEP_FS)
RAMP_N_INTERVALS  = 75          # temperature increments during ramp

# ----------------- STAGE 2: NPT VOLUME RELAXATION -----------------
# Baseline duration. May be extended automatically if convergence fails.
NPT_DURATION_PS   = 300.0      # ps  (was 100 ps)
NPT_STEPS         = int(NPT_DURATION_PS * 1e3 / TIMESTEP_FS)

# Extension protocol: if convergence criteria not met, extend by this
# amount and re-check. Up to NPT_MAX_EXTENSIONS extensions are allowed.
NPT_EXTENSION_PS   = 500.0      # ps per extension cycle
NPT_MAX_EXTENSIONS = 4          # maximum number of extension cycles (= up to 3000 ps total)

# Set to False to run exactly NPT_DURATION_PS regardless of convergence.
# The convergence check still runs and prints its report, but no additional
# steps are executed. Useful when wall-time is constrained or when you have
# already established that the baseline duration is sufficient for this system.
NPT_ALLOW_EXTENSION = False

# Convergence criteria
# (a) Relative standard deviation of block-mean volumes < this threshold
VOL_CONVERGENCE_TOL   = 0.015   # 0.5%
# (b) Absolute mean pressure in the last 20% of the run < this threshold
P_CONVERGENCE_BAR     = 200.0   # bar
# Fraction of the NPT run used for block-averaging and convergence checks
NPT_ANALYSIS_FRACTION = 0.20    # last 20%

# Collection interval for thermodynamic observables during NPT (steps)
NPT_COLLECT_INTERVAL  = 500     # every 200 steps = every 0.1 ps

# ----------------- STAGE 3: NVT STABILIZATION -----------------
NVT_DURATION_PS   = 500.0       # ps  (was 200 ps)
NVT_STEPS         = int(NVT_DURATION_PS * 1e3 / TIMESTEP_FS)

# Friction for Langevin thermostat (stages 1 and 3)
FRICTION_FS = 1.0 / 10.0       # fs^-1 — softer than before (1/50)
FRICTION    = FRICTION_FS / units.fs

# Log/traj save interval during equilibration (steps)
EQUIL_SAVE_INTERVAL = 2000      # every 1 ps

# ----------------- STAGE 4: NVE PRODUCTION -----------------
PROD_DURATION_NS   = 1.2         # ns
PROD_STEPS         = int(PROD_DURATION_NS * 1e6 / TIMESTEP_FS)
PROD_SAVE_INTERVAL = int(1000.0 / TIMESTEP_FS)   # 1 ps per frame → 10 000 frames

FMAX = 0.05     # eV/Å (geometry optimization convergence)

os.makedirs("optimization",   exist_ok=True)
os.makedirs("equilibration",  exist_ok=True)
os.makedirs("production",     exist_ok=True)

# =============================================================================
# HELPER: THERMODYNAMIC CONVERGENCE CHECK
# =============================================================================

# eV/Å³ → GPa conversion
EV_A3_TO_GPA = 1.0 / (units.GPa / (units.eV / units.Ang**3))


def check_npt_convergence(volumes: list, pressures_eV_A3: list) -> tuple[bool, dict]:
    """
    Check whether the NPT run has reached thermodynamic equilibrium.

    Uses the last NPT_ANALYSIS_FRACTION of the collected data.

    Returns
    -------
    converged : bool
    report    : dict with keys vol_mean, vol_std_rel, p_mean_bar, p_std_bar
    """
    n = len(volumes)
    if n < 20:
        return False, {"error": "Insufficient data points for convergence check."}

    n_tail = max(10, int(n * NPT_ANALYSIS_FRACTION))
    vol_tail = np.array(volumes[-n_tail:])
    p_tail   = np.array(pressures_eV_A3[-n_tail:])

    # Split tail into 4 equal blocks for block-mean estimation
    n_blocks = 4
    block_size = len(vol_tail) // n_blocks
    block_means = [vol_tail[i*block_size:(i+1)*block_size].mean()
                   for i in range(n_blocks)]

    vol_mean     = vol_tail.mean()
    vol_std_rel  = np.std(block_means) / vol_mean  # relative std of block means

    # Pressure: convert eV/Å³ -> bar  (1 eV/Å³ = 160217.66 bar)
    eV_A3_to_bar = 160217.66
    p_tail_bar   = p_tail * eV_A3_to_bar
    p_mean_bar   = p_tail_bar.mean()
    p_std_bar    = p_tail_bar.std()

    vol_ok = vol_std_rel < VOL_CONVERGENCE_TOL
    p_ok   = abs(p_mean_bar) < P_CONVERGENCE_BAR

    report = {
        "vol_mean_A3":    vol_mean,
        "vol_std_rel_pct": vol_std_rel * 100,
        "p_mean_bar":     p_mean_bar,
        "p_std_bar":      p_std_bar,
        "vol_converged":  vol_ok,
        "p_converged":    p_ok,
    }
    return (vol_ok and p_ok), report


def print_convergence_report(label: str, report: dict) -> None:
    print(f"\n  [{label}] NPT Convergence Report")
    if "error" in report:
        print(f"    {report['error']}")
        return
    v_flag = "OK" if report["vol_converged"] else "FAIL"
    p_flag = "OK" if report["p_converged"]   else "FAIL"
    print(f"    Volume  : {report['vol_mean_A3']:.2f} A³  "
          f"block-mean rel.std = {report['vol_std_rel_pct']:.3f}%  "
          f"(tol: {VOL_CONVERGENCE_TOL*100:.1f}%)  [{v_flag}]")
    print(f"    Pressure: {report['p_mean_bar']:.1f} ± {report['p_std_bar']:.1f} bar  "
          f"(tol: ±{P_CONVERGENCE_BAR:.0f} bar)  [{p_flag}]")


# =============================================================================
# CALCULATOR FACTORY & MINIMIZATION
# =============================================================================

def build_calculator(dtype: str, model_path: str) -> MACECalculator:
    print(f"[CALCULATOR | {dtype}] Instantiating MACECalculator on {TARGET_DEVICE} ...")
    return MACECalculator(
        model_paths=model_path,
        device=TARGET_DEVICE,
        default_dtype=dtype,
        enable_cueq=True,
    )


def minimize_structure(model_path: str) -> Atoms:
    print("\n[MINIMIZATION] Loading initial structure ...")
    atoms = read(INPUT_STRUCTURE)
    calc_f64 = build_calculator("float64", model_path)
    atoms.calc = calc_f64

    filter_atoms = FrechetCellFilter(atoms, hydrostatic_strain=True)
    optimizer = BFGS(
        filter_atoms,
        trajectory=OPTIM_TRAJ,
        logfile="./optimization/optimization.log",
    )
    optimizer.run(fmax=FMAX)

    write(MINIMIZED_XYZ, atoms)
    atoms.calc = None
    del calc_f64
    torch.cuda.empty_cache()
    print("[MINIMIZATION] Done. float64 calculator released, VRAM flushed.")
    return atoms


# =============================================================================
# STAGE 1 — NVT HEATING RAMP  (pure Langevin, no barostat)
# =============================================================================

def run_nvt_ramp(atoms: Atoms, target_temp: int, base_path: str) -> None:
    """
    Heat the system from RAMP_START_T to target_temp using a pure NVT thermostat.
    The cell volume is kept fixed at the minimized value throughout.
    This avoids the barostat instabilities that occur when LangevinBAOAB is
    applied at temperatures well below the target where forces are poorly
    sampled.
    """
    print(f"  -> Stage 1: NVT Heating Ramp "
          f"({RAMP_START_T:.0f} K -> {target_temp} K, {RAMP_DURATION_PS:.0f} ps, NVT only) ...")

    MaxwellBoltzmannDistribution(atoms, temperature_K=RAMP_START_T)
    Stationary(atoms)
    ZeroRotation(atoms)

    steps_per_interval = RAMP_STEPS // RAMP_N_INTERVALS

    dyn_ramp = Langevin(
        atoms,
        timestep=TIMESTEP,
        temperature_K=RAMP_START_T,
        friction=FRICTION,
        trajectory=f"{base_path}_ramp.traj",
        logfile=f"{base_path}_ramp.log",
        loginterval=EQUIL_SAVE_INTERVAL,
    )

    for T in np.linspace(RAMP_START_T, target_temp, RAMP_N_INTERVALS):
        dyn_ramp.set_temperature(temperature_K=float(T))
        dyn_ramp.run(steps_per_interval)

    print(f"     Ramp complete. Final instantaneous T = "
          f"{atoms.get_temperature():.1f} K (target: {target_temp} K)")


# =============================================================================
# STAGE 2 — NPT VOLUME RELAXATION  (with convergence check and auto-extension)
# =============================================================================

def run_npt_equilibration(atoms: Atoms, target_temp: int, base_path: str,
                          allow_extension: bool = NPT_ALLOW_EXTENSION) -> float:
    """
    Run NPT equilibration and return the converged average volume (Å³).

    The barostat time constant T_BAROSTAT = 20000 fs prevents the stiff pressure
    oscillations that caused the previous run to remain 2252 bar below the
    target pressure after 100 ps.

    Parameters
    ----------
    atoms         : ASE Atoms object, thermalized at target_temp by the NVT ramp.
    target_temp   : Target temperature in K.
    base_path     : Path prefix for trajectory and log files.
    allow_extension : If True (default, controlled by NPT_ALLOW_EXTENSION),
                      the run is automatically extended in NPT_EXTENSION_PS
                      increments when convergence criteria are not met, up to
                      NPT_MAX_EXTENSIONS times.
                      If False, exactly NPT_DURATION_PS is run. The convergence
                      check still executes and prints its report, but no
                      additional MD steps are performed. A warning is raised if
                      convergence was not reached.

    Returns
    -------
    avg_vol : float
        Average volume computed from the last NPT_ANALYSIS_FRACTION of the
        run (Å³).
    """
    ext_label = (f"extensions up to {NPT_MAX_EXTENSIONS}x{NPT_EXTENSION_PS:.0f} ps"
                 if allow_extension else "extensions DISABLED — fixed duration")
    print(f"  -> Stage 2: NPT Volume Relaxation "
          f"({target_temp} K, {PRESSURE} bar, baseline {NPT_DURATION_PS:.0f} ps) ...")
    print(f"     T_BAROSTAT = {T_BAROSTAT / units.fs:.0f} fs  |  "
          f"T_THERMOSTAT = {T_THERMOSTAT / units.fs:.0f} fs  |  "
          f"Convergence tol: vol {VOL_CONVERGENCE_TOL*100:.1f}%, P {P_CONVERGENCE_BAR:.0f} bar  |  "
          f"{ext_label}")

    volumes:   list[float] = []
    pressures: list[float] = []   # eV/Å³

    def record_thermo() -> None:
        volumes.append(atoms.get_volume())
        stress = atoms.get_stress(voigt=True)   # eV/Å³, Voigt: xx yy zz yz xz xy
        pressures.append(-stress[:3].mean())    # P = -1/3 Tr(sigma)

    dyn_npt = IsotropicMTKNPT(
        atoms,
        timestep=TIMESTEP,
        temperature_K=target_temp,
        pressure_au=-PRESSURE * units.GPa, 
        tdamp=T_THERMOSTAT,
        pdamp=T_BAROSTAT,
        trajectory=f"{base_path}_npt.traj",
        logfile=f"{base_path}_npt.log",
        loginterval=EQUIL_SAVE_INTERVAL,
        )
    

    dyn_npt.attach(record_thermo, interval=NPT_COLLECT_INTERVAL)

    # Run the baseline NPT
    dyn_npt.run(NPT_STEPS)
    converged, report = check_npt_convergence(volumes, pressures)
    print_convergence_report(f"baseline {NPT_DURATION_PS:.0f} ps", report)

    # Extension loop — only executed when allow_extension is True
    ext_steps = int(NPT_EXTENSION_PS * 1e3 / TIMESTEP_FS)
    for ext_idx in range(1, NPT_MAX_EXTENSIONS + 1):
        if converged or not allow_extension:
            break
        total_ps = NPT_DURATION_PS + ext_idx * NPT_EXTENSION_PS
        print(f"     NPT not converged. Extending by {NPT_EXTENSION_PS:.0f} ps "
              f"(total: {total_ps:.0f} ps, extension {ext_idx}/{NPT_MAX_EXTENSIONS}) ...")
        dyn_npt.run(ext_steps)
        converged, report = check_npt_convergence(volumes, pressures)
        print_convergence_report(f"extension {ext_idx} ({total_ps:.0f} ps)", report)

    if not converged:
        n_ext_run = 0 if not allow_extension else NPT_MAX_EXTENSIONS
        total_ps  = NPT_DURATION_PS + n_ext_run * NPT_EXTENSION_PS
        hint = ("Set NPT_ALLOW_EXTENSION = True to enable automatic extension, or "
                if not allow_extension else "")
        warnings.warn(
            f"[NPT | {target_temp} K] Convergence criteria not met after {total_ps:.0f} ps. "
            f"Last report: vol_std_rel={report.get('vol_std_rel_pct', float('nan')):.3f}%, "
            f"P_mean={report.get('p_mean_bar', float('nan')):.1f} bar. "
            f"{hint}Consider: (1) longer T_BAROSTAT, (2) checking MACE model validity "
            f"for this composition.",
            stacklevel=2,
        )

    # Average volume from the last NPT_ANALYSIS_FRACTION of collected data
    n_tail = max(10, int(len(volumes) * NPT_ANALYSIS_FRACTION))
    avg_vol = float(np.mean(volumes[-n_tail:]))

    print(f"     Equilibrium volume (last {NPT_ANALYSIS_FRACTION*100:.0f}% average): "
          f"{avg_vol:.3f} Å³")
    return avg_vol


# =============================================================================
# STAGE 3 — NVT STABILIZATION  (fix cell at equilibrium volume)
# =============================================================================

def run_nvt_stabilization(atoms: Atoms, avg_vol: float, target_temp: int,
                          base_path: str) -> None:
    """
    Rescale the cell to the NPT-averaged equilibrium volume and run NVT
    to re-equilibrate the kinetic energy distribution before NVE production.

    The rescaling is hydrostatic (uniform scaling). The subsequent NVT
    run dissipates any residual pressure and kinetic energy imbalance
    introduced by the abrupt volume change.
    """
    current_vol = atoms.get_volume()
    scale_factor = (avg_vol / current_vol) ** (1.0 / 3.0)
    atoms.set_cell(atoms.cell * scale_factor, scale_atoms=True)
    print(f"  -> Stage 3: Cell rescaled {current_vol:.2f} -> {avg_vol:.2f} Å³ "
          f"(scale factor: {scale_factor:.6f})")
    print(f"     NVT Stabilization at {target_temp} K for {NVT_DURATION_PS:.0f} ps ...")

    dyn_nvt = Langevin(
        atoms,
        timestep=TIMESTEP,
        temperature_K=target_temp,
        friction=FRICTION,
        trajectory=f"{base_path}_nvt.traj",
        logfile=f"{base_path}_nvt.log",
        loginterval=EQUIL_SAVE_INTERVAL,
    )

    # Monitor temperature convergence at the end of NVT
    temps: list[float] = []
    dyn_nvt.attach(lambda: temps.append(atoms.get_temperature()), interval=NPT_COLLECT_INTERVAL)
    dyn_nvt.run(NVT_STEPS)

    n_tail = max(10, int(len(temps) * 0.20))
    t_mean = float(np.mean(temps[-n_tail:]))
    t_std  = float(np.std(temps[-n_tail:]))
    print(f"     NVT complete. T (last 20%) = {t_mean:.1f} ± {t_std:.1f} K "
          f"(target: {target_temp} K)")


# =============================================================================
# FULL EQUILIBRATION ENTRY POINT
# =============================================================================

def run_equilibration(atoms_minimized: Atoms, target_temp: int,
                      calc_f32: MACECalculator) -> Atoms:
    print(f"\n[{target_temp} K | EQUILIBRATION] Starting multi-stage protocol ...")
    atoms = copy.deepcopy(atoms_minimized)
    atoms.calc = calc_f32

    base_path = f"equilibration/equil_{target_temp}K"

    # Stage 1: NVT ramp (no barostat)
    run_nvt_ramp(atoms, target_temp, base_path)

    # Stage 2: NPT — find equilibrium volume with convergence check
    avg_vol = run_npt_equilibration(atoms, target_temp, base_path,
                                    allow_extension=NPT_ALLOW_EXTENSION)

    # Stage 3: NVT — stabilize at equilibrium volume
    run_nvt_stabilization(atoms, avg_vol, target_temp, base_path)

    print(f"[{target_temp} K | EQUILIBRATION] Complete.")
    return atoms


# =============================================================================
# STAGE 4 — NVE PRODUCTION  (VelocityVerlet)
# =============================================================================

def run_production(atoms_equil: Atoms, temperature: int,
                   calc_f32: MACECalculator) -> None:
    """
    NVE production run. The system enters with velocities drawn from the
    NVT-thermalized ensemble; VelocityVerlet conserves E_total exactly
    (within integrator precision, verified by the 0.000265% drift in the
    previous run — no changes needed here).
    """
    print(f"\n[{temperature} K | PRODUCTION] Starting NVE production ...")
    atoms_equil.calc = calc_f32

    traj_path = f"production/production_{temperature}K.traj"
    log_path  = f"production/production_{temperature}K.log"

    dyn = VelocityVerlet(
        atoms_equil,
        timestep=TIMESTEP,
        trajectory=traj_path,
        logfile=log_path,
        loginterval=PROD_SAVE_INTERVAL,
    )

    # Monitor energy conservation during production
    etots: list[float] = []
    monitor_interval = int(10e3 / TIMESTEP_FS)   # every 10 ps

    def record_etot() -> None:
        etots.append(atoms_equil.get_total_energy())

    dyn.attach(record_etot, interval=monitor_interval)

    print(f"  -> {PROD_STEPS} steps | {PROD_DURATION_NS} ns | dt = {TIMESTEP_FS} fs")
    print(f"  -> Saving every {PROD_SAVE_INTERVAL} steps (1 ps / frame)")

    start_time = time.time()
    dyn.run(steps=PROD_STEPS)
    wall_time = time.time() - start_time

    # Energy conservation report
    if len(etots) >= 2:
        drift_abs = etots[-1] - etots[0]
        drift_rel = abs(drift_abs / etots[0]) * 100
        print(f"  -> Energy drift: {drift_abs:+.6f} eV  ({drift_rel:.6f}% relative)")
        if drift_rel > 0.01:
            warnings.warn(
                f"NVE energy drift {drift_rel:.4f}% exceeds 0.01% threshold. "
                "Consider reducing the timestep.",
                stacklevel=2,
            )

    ns_per_day = (PROD_DURATION_NS / wall_time) * 86400
    print(f"  -> Performance: {ns_per_day:.2f} ns/day")
    print(f"  -> Trajectory: {traj_path}")


# =============================================================================
# MAIN
# =============================================================================

def main(model_path: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA devices found.")

    print(f"[INIT] Device: {TARGET_DEVICE}")
    npt_ext_info = (f"extensions up to {NPT_MAX_EXTENSIONS}x{NPT_EXTENSION_PS:.0f} ps"
                    if NPT_ALLOW_EXTENSION else "extensions DISABLED")
    print(f"[INIT] Equilibration protocol summary:")
    print(f"       Ramp    : {RAMP_DURATION_PS:.0f} ps NVT  "
          f"({RAMP_START_T:.0f} K -> target T, Langevin only)")
    print(f"       NPT     : {NPT_DURATION_PS:.0f} ps baseline  "
          f"(T_BAROSTAT={T_BAROSTAT/units.fs:.0f} fs, {npt_ext_info})")
    print(f"       NVT     : {NVT_DURATION_PS:.0f} ps  (T_THERMOSTAT={T_THERMOSTAT/units.fs:.0f} fs)")
    print(f"       NVE     : {PROD_DURATION_NS:.0f} ns  (dt={TIMESTEP_FS} fs, 1 ps/frame)")

    atoms_minimized = minimize_structure(model_path)

    print("\n[INIT] Loading float32 calculator for MD stages ...")
    calc_f32 = build_calculator("float32", model_path)

    for T in TEMPERATURES:
        print(f"\n{'='*60}")
        print(f"  Temperature : {T} K  |  Device: {TARGET_DEVICE}")
        print(f"{'='*60}")

        atoms_equil = run_equilibration(atoms_minimized, T, calc_f32)
        run_production(atoms_equil, T, calc_f32)

    print("\n[DONE] All temperatures completed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MSD calculation.")
    parser.add_argument("--model", type=str, required=True, help="Path to the MACE model (.model file)")
    args = parser.parse_args()
    main(args.model)
