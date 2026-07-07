[![DOI](https://zenodo.org/badge/997232519.svg)](https://doi.org/10.5281/zenodo.17699011)

# MACE Workflow

A research repository for **MACE** machine learning force fields applied to zeolite and perovskite materials.

## Repository Structure

### 📊 Article Calculations (`article-calculations/`)

Research calculations and simulations for specific zeolite structures.

#### AFI Zeolite
- **MD simulations**
- **Linearity minimization**
- **Thermobarostat benchmarking**
- **NPT constant benchmarking**
- **O-Si-O angle calculations**
- Structural files in VASP and XYZ formats

#### FAU Zeolite
- **Thermal expansion**
- **Timestep benchmarking**
- Optimization and MD simulation scripts

#### MFI Zeolite
- **Phase comparison** (monoclinic vs orthorhombic)
- **Energy comparisons**
- **Temperature ramp simulations**
- Phase transition studies
- LAMMPS and VASP structure files

#### RHO Zeolite
- **Pressure-dependent minimization**
- **E+pV calculations**
- Multiple pressure point directories with structural data

### 🛠️ Utility Functions (`functions/`)

Reusable tools for data processing, analysis, and VASP simulation setup:

- **Structure Conversion**: `atoms_utils.py`, `convert_stress_units.py`
- **Dataset Generation**: `generate_dataset.py` - creates training datasets from VASP files with various sampling strategies
- **XYZ File Manipulation**: 
  - `split_xyz_files.py` - split XYZ files into train/test sets
  - `sort_energies.py` - sort structures by energy
  - `show-xyz-properties.py` - visualize properties from XYZ files
- **Filtering Utilities**:
  - `filter_by_element.py` - filter structures by atomic composition
  - `remove_structures_with_element.py` - filter out frames containing a specific chemical element
  - `filter_forces.py` - filter by force magnitudes
  - `filter_properties.py` - general property filtering
- **Data Validation & Topology**:
  - `check_si_o_ratio.py` - validate Si/O ratios in zeolites
  - `review-broken-frames.py` - identify problematic structures
  - `sio2_bond_checker.py` - count unique bonds and compare bond topology between two SiO2/zeolite structures within a distance cutoff
- **MD & Trajectory Analysis**:
  - `ase-rdf.py` - compute pair Radial Distribution Functions (RDF) and coordination numbers $N(r)$ from ASE-readable trajectories with an adaptive minimum-image convention constraint on $r_{max}$
  - `log_mdmean.py` - analyze text-based molecular dynamics logs, automatically detecting equilibration time $t_0$, computing block averages, and plotting statistical parameters
  - `traj_mdmean.py` - analyze ASE-compatible trajectory files (Potential/Kinetic/Total energy, Temperature, Volume, Cell parameters, Pressure) with equilibration detection and block averages
- **Simulation Workflows**:
  - `run_msd.py` - multi-stage workflow script for vacancy-mediated ionic diffusion in hybrid perovskites (incorporates geometry optimization, NVT heating ramp, NPT volume relaxation with convergence checks, NVT stabilization, and NVE production using MACE)
- **VASP Utilities**:
  - `extract.sh` - build consolidated XYZ files from VASP XML outputs and generate splits
  - `generate_potcar.sh` - generate VASP POTCAR files for POSCAR species
  - `generate_potcar_soft.sh` - generate VASP POTCAR files using soft potentials (`C_s`, `N_s`, `H_s`)
- **Visualization**: 
  - `plot_mace_mlff.py` - plot MACE predictions
  - `plot_predictions.py` - compare predictions vs reference
  - `final-plot-preds.py` -


### 🎓 Training Configurations (`trainings/`)

Machine learning model training setups:

- **Zeolites**: 
  - Fine-tuning configurations (source-closed and source-open)
  - Data preprocessing configurations
  - Input structures and training datasets
- **Perovskites**: Training configurations for perovskite materials

### 📚 Tutorials

PDF documentation for MACE usage available in multiple languages.

### MD Ensembles Supported
- NPT (Berendsen, Martyna-Tobias-Klein)
- NVT (Langevin, Nose-Hoover Chain)
- Energy minimization with various optimizers (BFGS, FIRE)

## Author

Antonio Rivas (@RivasAntonio)

