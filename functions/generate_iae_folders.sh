#!/bin/bash
# ==============================
# Script para generar cálculos de átomo aislado
# Uso: bash crear_isolated_atoms.sh H C N O Si Br I Pb
# ==============================

# Ruta a los POTCAR
potcar_base=~/software/VASP/data

# MAGMOM inicial recomendado por elemento
declare -A magmom_dict=(
  [H]=1.0
  [C]=2.0
  [N]=3.0
  [O]=2.0
  [Si]=2.0
  [Br]=1.0
  [I]=1.0
  [Pb]=2.0
)

# Crear directorio por elemento
for elem in "$@"; do
  dir="dir_${elem}"
  mkdir -p "$dir"

  echo "Creando cálculo para $elem en $dir"

  # ===== POSCAR =====
  cat > "$dir/POSCAR" <<EOF
${elem} isolated atom
1.0
15.0 0.0 0.0
0.0 15.1 0.0
0.0 0.0 15.2
${elem}
1
Direct
0.5 0.5 0.5
EOF
  
#
# The primary reason for using a slightly asymmetric box (a nearly cubic cell with minimal orthorhombic distortion) 
# is to artificially break the spatial symmetry of the atom's environment. In a perfectly symmetric cubic box, certain atomic orbitals 
# (like the p and d orbitals) are perfectly degenerate, meaning they have the exact same energy level. For atoms with partially filled 
# valence shells, such as Carbon, Nitrogen, or transition metals, this perfect degeneracy causes problems for the self-consistent field (SCF) 
# cycle. This geometric distortion slightly splits the energy levels of the p and d orbitals, removing the perfect degeneracy.
# 

  # ===== KPOINTS =====
  cat > "$dir/KPOINTS" <<EOF
Gamma-point
0
Gamma
1 1 1
0 0 0
EOF

  # ===== POTCAR =====
  potcar_path=$(find "$potcar_base" -maxdepth 1 -type d -name "${elem}*" | head -n 1)
  if [[ -d "$potcar_path" && -f "$potcar_path/POTCAR" ]]; then
    cp "$potcar_path/POTCAR" "$dir/"
  else
    echo "⚠️  No se encontró POTCAR para ${elem} en ${potcar_base}"
    continue
  fi

  # ===== Calcular ENCUT = 1.4 * ENMAX =====
  raw_enmax=$(grep -m1 "ENMAX" "$dir/POTCAR" | awk '{print $3}')
  # eliminar cualquier carácter no numérico (como ';')
  enmax=$(echo "$raw_enmax" | tr -d ';')
  encut=$(awk -v e="$enmax" 'BEGIN {printf "%.1f", e * 1.4}')

  # ===== INCAR =====
  magmom=${magmom_dict[$elem]:-1.0}
  cat > "$dir/INCAR" <<EOF
# ======= Isolated atom calculation =======
SYSTEM = ${elem} isolated atom
ISTART = 0
ICHARG = 2
ISPIN  = 2
MAGMOM = ${magmom}

# ======= Functional: r2SCAN + rVV10 =======
METAGGA = r2SCAN
LUSE_VDW = .TRUE.
BPARAM = 15.7
CPARAM = 0.0093

# ======= Cutoff and precision =======
ENCUT = 520
PREC = Accurate

# ======= Electronic minimization =======
EDIFF = 1E-6
ISMEAR = 0
SIGMA = 0.01
NELM = 200

# ======= Ionic settings =======
IBRION = -1
NSW = 0

# ======= Output =======
ISYM = 0
LREAL = .FALSE.
LWAVE = .TRUE.
LCHARG = .TRUE.
LASPH =.TRUE.
LMAXMIX = 6
LMIXTAU =.TRUE.
EOF

  # ===== SLURM SCRIPT =====
  cat > "$dir/submit.slurm" <<EOF
#!/bin/bash
#SBATCH -o slurm_out.log
#SBATCH -e slurm_error.log
#SBATCH -t 10-00:00:00
#SBATCH -p standard
#SBATCH -J iae_${elem}
#SBATCH --mem=180G
#SBATCH --nodes=2             
#SBATCH --ntasks=96           
#SBATCH --ntasks-per-node=48  
#SBATCH --cpus-per-task=1     

export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
ulimit -s unlimited
module load FFTW.MPI/3.3.10-gompi-2023a ScaLAPACK/2.2.0-gompi-2023a-fb HDF5/1.14.0-gompi-2023a

# Archivo de log para el progreso
LOG_FILE="progreso_${elem}.log"

echo "Cálculo de átomo aislado para ${elem}" > "\$LOG_FILE"
echo "Inicio: \$(date)" >> "\$LOG_FILE"
echo "--------------------------------------------" >> "\$LOG_FILE"

echo "[\$(date)] Iniciando cálculo VASP para ${elem}" | tee -a "\$LOG_FILE"

# Ejecutar VASP
mpirun --bind-to core:overload-allowed --map-by socket -np \$SLURM_NTASKS --report-bindings \${HOME}/software/VASP/vasp.6.5.0/bin/vasp_std

# Limpiar archivos innecesarios
rm -f WAVECAR CH*

echo "[\$(date)] Cálculo completado para ${elem}" | tee -a "\$LOG_FILE"
EOF

done
