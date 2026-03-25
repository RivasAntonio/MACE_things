#!/bin/bash
# SLURM job script to run VASP single-point energy calculations.

#==============================================================================
# CONFIGURACIÓN SLURM - Recursos del cluster
#==============================================================================
#SBATCH --job-name=vasp_singlepoint               # Nombre del trabajo
#SBATCH -o general_out.out                         # Archivo de salida estándar
#SBATCH -e general_error.out                       # Archivo de errores
#SBATCH -t 5-00:00:00                              # Tiempo máximo: 5 días
#SBATCH -p standard                                # Partición/cola a usar
#SBATCH --nodes=1                                  # Número de nodos
#SBATCH --ntasks=48                                # Número total de tareas MPI
#SBATCH --ntasks-per-node=48                       # Tareas por nodo
#SBATCH --cpus-per-task=1                          # CPUs por tarea

#==============================================================================
# VARIABLES DE ENTORNO Y CONFIGURACIÓN
#==============================================================================
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

LOG_FILE="singlepoint_${SLURM_JOB_ID}.log"

# Función de logging con timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Extrae energía libre (TOTEN) del OSZICAR
get_free_energy() {
    grep "F=" OSZICAR | tail -n1 | awk '{print $3}'
}

# Extrae energía sin entropía (E0) del OSZICAR
get_energy_e0() {
    grep "F=" OSZICAR | tail -n1 | awk '{print $5}'
}

# Extrae fuerza máxima del OUTCAR
get_max_force() {
    grep "FORCES: max atom" OUTCAR | tail -n1 | awk '{print $5}' 2>/dev/null || echo "N/A"
}

# Limpia archivos pesados innecesarios
clean_heavy_files() {
    rm -f CHG IBZKPT PCDAT XDATCAR REPORT core* 2>/dev/null
    log_message "Archivos pesados eliminados"
}

#==============================================================================
# CARGA DE MÓDULOS
#==============================================================================
module purge
module load FFTW.MPI/3.3.10-gompi-2023a ScaLAPACK/2.2.0-gompi-2023a-fb HDF5/1.14.0-gompi-2023a

#==============================================================================
# REGISTRO DE JOB EN ARCHIVO DE TRACKING
#==============================================================================
mkdir -p ~/jobs_running
echo "$SLURM_JOB_ID $(pwd)" >> ~/jobs_running/running_vasp
trap "sed -i '/$SLURM_JOB_ID/d' ~/jobs_running/running_vasp" EXIT

#==============================================================================
# CONFIGURACIÓN DE VASP
#==============================================================================
export VASP_DIR=/lustre/home/arivas/software/VASP/vasp.6.5.0/bin
cp /lustre/home/arivas/software/VASP/data/vdw_kernel.bindat .

NUMCORES_PER_BAND=6
log_message "Configuración de paralelización: NCORE=$NUMCORES_PER_BAND para $SLURM_NTASKS cores"

#==============================================================================
# PREPARACIÓN DEL DIRECTORIO DE SCRATCH
#==============================================================================
export SCRATCH=/lustre/scratch/jobs/$USER/$SLURM_JOB_ID

log_message "Creando directorio temporal $SCRATCH"
mkdir -p $SCRATCH || exit $?

log_message "Copiando archivos a $SCRATCH"
echo $SLURM_SUBMIT_DIR > original_dir_$SLURM_JOB_ID.txt
cp -r $SLURM_SUBMIT_DIR/* $SCRATCH || exit $?
echo "/lustre/scratch/jobs/$USER/$SLURM_JOB_ID" > running_dir_$SLURM_JOB_ID.txt

mkdir ZZ_tmp_input
mv * ZZ_tmp_input
mkdir bckup

cd $SCRATCH

#==============================================================================
# BACKUP DE ARCHIVOS DE ENTRADA
#==============================================================================
log_message "Creando backup de archivos de entrada"
cp KPOINTS POTCAR POSCAR INCAR bckup/

#==============================================================================
# PREPARACIÓN DEL INCAR PARA SINGLE POINT
#==============================================================================
log_message "Configurando INCAR para cálculo de punto único"

# Fuerza NSW=0 (sin ciclos de optimización)
sed -i 's/^\s*NSW\s*=.*/NSW = 0  # Single point calculation/' INCAR
grep -q "^NSW" INCAR || echo "NSW = 0  # Single point calculation" >> INCAR

# Asegura NCORE para paralelización eficiente
grep -q "NCORE" INCAR || echo "NCORE = $NUMCORES_PER_BAND  # Optimización paralelización" >> INCAR

#==============================================================================
# EJECUCIÓN DE VASP - CÁLCULO DE PUNTO ÚNICO
#==============================================================================
log_message "========================================="
log_message "Iniciando cálculo de punto único"
log_message "========================================="

time mpirun --bind-to core:overload-allowed --map-by socket -np $SLURM_NTASKS --report-bindings ${VASP_DIR}/vasp_std
vasp_exit=$?
rm -f core* 2>/dev/null

# Validación del resultado
if [ $vasp_exit -ne 0 ]; then
    log_message "ERROR: VASP falló con código $vasp_exit"
    exit 1
fi

if [ ! -s OUTCAR ]; then
    log_message "ERROR: OUTCAR vacío o no generado"
    exit 1
fi

# Extrae y reporta resultados
free_energy=$(get_free_energy)
energy_e0=$(get_energy_e0)
max_force=$(get_max_force)

log_message "========================================="
log_message "RESULTADOS:"
log_message "  Energía libre (F):    $free_energy eV"
log_message "  Energía sin entropía (E0): $energy_e0 eV"
log_message "  Fuerza máxima:        $max_force eV/Å"
log_message "========================================="

# Verifica convergencia SCF
if grep -q "aborting loop because EDIFF is reached" OUTCAR; then
    log_message "✓ SCF convergido (EDIFF alcanzado)"
else
    log_message "⚠ ADVERTENCIA: SCF puede no haber convergido. Revise OUTCAR."
fi

#==============================================================================
# BACKUP DE RESULTADOS
#==============================================================================
log_message "Guardando resultados en bckup/"
cp CONTCAR bckup/
cp OSZICAR bckup/
cp OUTCAR  bckup/
[ -f vasprun.xml ] && cp vasprun.xml bckup/
[ -f CHGCAR ]     && cp CHGCAR bckup/
[ -f WAVECAR ]    && cp WAVECAR bckup/

clean_heavy_files

#==============================================================================
# LIMPIEZA Y COPIA FINAL DE RESULTADOS
#==============================================================================
cd $SLURM_SUBMIT_DIR
log_message "Copiando resultados finales a $SLURM_SUBMIT_DIR"
cp -r ${SCRATCH}/* ${SLURM_SUBMIT_DIR} || exit $?
mv ZZ_tmp_input/* .
log_message "Eliminando directorio temporal $SCRATCH"
rm -rf $SCRATCH || exit $?
rmdir ZZ_tmp_input 2>/dev/null || true

log_message "========================================="
log_message "CÁLCULO DE PUNTO ÚNICO COMPLETADO"
log_message "========================================="
