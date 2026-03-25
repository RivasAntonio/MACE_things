#!/bin/bash
# SLURM job script to run VASP geometry relaxations.

#==============================================================================
# CONFIGURACIÓN SLURM - Recursos del cluster
#==============================================================================
#SBATCH --job-name=per_geom_opt_low_ediff           # Nombre del trabajo
#SBATCH -o general_out.out                # Archivo de salida estándar
#SBATCH -e general_error.out              # Archivo de errores
#SBATCH -t 30-00:00:00                    # Tiempo máximo: 30 días
#SBATCH -p standard                       # Partición/cola a usar
#SBATCH --nodes=1                         # Número de nodos
#SBATCH --ntasks=48                       # Número total de tareas MPI
#SBATCH --ntasks-per-node=48              # Tareas por nodo
#SBATCH --cpus-per-task=1                 # CPUs por tarea

#==============================================================================
# VARIABLES DE ENTORNO Y CONFIGURACIÓN
#==============================================================================
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Recursos ilimitados para memoria y stack

# Variables configurables
CONV_CYCLES=5           # Ciclos máximos para considerar convergencia
MAX_ITERATIONS=150      # Iteraciones máximas de seguridad
LOG_FILE="optimization_${SLURM_JOB_ID}.log"

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
    grep "FORCES: max atom" OUTCAR | tail -n1 | awk '{print $5}' 2>/dev/null || echo "999"
}

# Limpia archivos pesados innecesarios
clean_heavy_files() {
    rm -f  CHG  IBZKPT PCDAT XDATCAR  REPORT core* 2>/dev/null
    log_message "Archivos pesados eliminados (PROCAR, CHG, DOSCAR,  etc.)"
}

# Guarda checkpoint para recuperación
save_checkpoint() {
    local iter=$1
    echo "$iter" > ../checkpoint.dat
    log_message "Checkpoint guardado: iteración $iter"
}

# Función de validación de VASP
check_vasp_exit() {
    local exit_code=$1
    local step_name=$2
    if [ $exit_code -ne 0 ]; then
        log_message "ERROR: VASP falló con código $exit_code en $step_name"
        exit 1
    fi
    if [ ! -s CONTCAR ]; then
        log_message "ERROR: CONTCAR vacío o no generado en $step_name"
        exit 1
    fi
}

#==============================================================================
# CARGA DE MÓDULOS
#==============================================================================
module purge
module load FFTW.MPI/3.3.10-gompi-2023a ScaLAPACK/2.2.0-gompi-2023a-fb HDF5/1.14.0-gompi-2023a

#==============================================================================
# REGISTRO DE JOB EN ARCHIVO DE TRACKING
#==============================================================================
# Registra el ID del job y el directorio de trabajo para monitoreo
mkdir -p ~/jobs_running
echo "$SLURM_JOB_ID $(pwd)" >> ~/jobs_running/running_vasp
# Limpieza automática al finalizar (éxito o error)
trap "sed -i '/$SLURM_JOB_ID/d' ~/jobs_running/running_vasp" EXIT

#==============================================================================
# CONFIGURACIÓN DE VASP
#==============================================================================
export VASP_DIR=/lustre/home/arivas/software/VASP/vasp.6.5.0/bin
cp /lustre/home/arivas/software/VASP/data/vdw_kernel.bindat .  # Kernel para correcciones van der Waals

# Optimización de paralelización: NCORE = número de cores por banda
# Para 48 cores: NCORE=4-8 es óptimo (NCORE=sqrt(cores) aproximadamente)
NUMCORES_PER_BAND=6  # Ajustar según sistema (4-8 para 48 cores)
log_message "Configuración de paralelización: NCORE=$NUMCORES_PER_BAND para $SLURM_NTASKS cores"

#==============================================================================
# PREPARACIÓN DEL DIRECTORIO DE SCRATCH
#==============================================================================
export SCRATCH=/lustre/scratch/jobs/$USER/$SLURM_JOB_ID

echo "Creating temp dir $SCRATCH"
mkdir -p $SCRATCH || exit $?

log_message "Copying files to $SCRATCH"
echo $SLURM_SUBMIT_DIR > original_dir_$SLURM_JOB_ID.txt
cp -r $SLURM_SUBMIT_DIR/*  $SCRATCH || exit $?
echo "/lustre/scratch/jobs/$USER/$SLURM_JOB_ID" > running_dir_$SLURM_JOB_ID.txt

# Organiza archivos originales en directorio temporal
mkdir ZZ_tmp_input
mv * ZZ_tmp_input
mkdir savebckup

cd $SCRATCH

#==============================================================================
# BACKUP DE ARCHIVOS DE ENTRADA
#==============================================================================
log_message "Creando backup de archivos de entrada"
mkdir bckup
cp KPOINTS POTCAR POSCAR INCAR bckup
cp POSCAR POSCAR_ini
cp POSCAR_ini bckup

# Verifica si existe checkpoint para recuperación
if [ -f checkpoint.dat ]; then
    RESTART_ITER=$(cat checkpoint.dat)
    log_message "Checkpoint encontrado: Reiniciando desde iteración $RESTART_ITER"
fi

# Asegura que LWAVE = .TRUE. para guardar la función de onda (WAVECAR)
sed -i 's/LWAVE\ \=\ \.FALSE\./LWAVE\ \=\ \.TRUE\./' INCAR
sed -i 's/LWAVE\=\.FALSE\./LWAVE\=\.TRUE\./' INCAR
sed -i 's/LWAVE\ \=\ \.False\./LWAVE\ \=\ \.True\./' INCAR
sed -i 's/LWAVE\=\.False\./LWAVE\=\.True\./' INCAR

#==============================================================================
# SIMU_00: Generación inicial de WAVECAR (solo si no existe)
#==============================================================================
# Optimización: omite este paso si ya existe WAVECAR válido (warm restart)
if [ ! -f WAVECAR ] || [ ! -s WAVECAR ]; then
    log_message "No hay WAVECAR previo. Generando WAVECAR inicial..."
    mkdir SIMU_00
    cp KPOINTS POTCAR POSCAR INCAR vdw_kernel.bindat SIMU_00
    [ -f CHGCAR ] && cp CHGCAR SIMU_00/
    cd SIMU_00
       # Modifica INCAR para establecer NSW = 0 (sin ciclos de optimización)
       sed -i 's/NSW/nsw1nsw2text\nNSW/' INCAR
       sed -i 's/NSW/\#NSW/' INCAR
       sed -i 's/nsw1nsw2text/NSW    =       0   # No cycle, only for WAVECAR creation/' INCAR
       
       # Agrega NCORE para paralelización eficiente (si no existe)
       grep -q "NCORE" INCAR || echo "NCORE = $NUMCORES_PER_BAND  # Optimización paralelización" >> INCAR 
       
       # Ejecuta VASP en modo single-point
       log_message "Iniciando SIMU_00: generación inicial de WAVECAR"
       time mpirun --bind-to core:overload-allowed --map-by socket -np $SLURM_NTASKS --report-bindings ${VASP_DIR}/vasp_std
       vasp_exit=$?
       rm core* 2>/dev/null
       
       # Valida ejecución exitosa
       if [ $vasp_exit -ne 0 ]; then
           log_message "ERROR: VASP falló en SIMU_00 con código $vasp_exit"
           exit 1
       fi
       if [ ! -s WAVECAR ]; then
           log_message "ERROR: WAVECAR no generado en SIMU_00"
           exit 1
       fi
       
       # Mueve archivos generados al directorio padre
       mv WAVECAR ../
       mv CHGCAR ../ 2>/dev/null || true
       
       log_message "WAVECAR y CHGCAR generados exitosamente"
    cd ../
    # Limpia archivos innecesarios de SIMU_00
    cd SIMU_00
    clean_heavy_files
    cd ..
else
    log_message "✓ WAVECAR existente encontrado. Saltando SIMU_00 (ahorro de tiempo)"
fi

#==============================================================================
# CICLO DE OPTIMIZACIÓN AUTOMÁTICA (SIMU_01, SIMU_02, ...)
#==============================================================================
# Ejecuta VASP iterativamente hasta que la geometría converja
# Criterio de convergencia: última corrida con ≤3 pasos de optimización

vaspOK=0      # Flag de convergencia (0 = no converge, 1 = converge)
count1=0      # Contador de iteraciones

while [ $vaspOK -eq 0 ]
do
  count1=$(( count1+1 ))
  
  # Define nombre del directorio actual (SIMU_01, SIMU_02, ..., SIMU_10, ...)
  if [ $count1 -le 9 ]
     then
       currentdir=SIMU_0$count1
     else
       currentdir=SIMU_$count1
  fi
  
  log_message "========================================="
  log_message "Iniciando iteración $count1: $currentdir"
  log_message "========================================="
  
  mkdir $currentdir
  cd $currentdir
    # Copia archivos de entrada (usa CONTCAR previo como nuevo POSCAR)
    # Incluye CHGCAR para acelerar convergencia SCF
    cp ../KPOINTS ../POTCAR ../POSCAR ../INCAR ../WAVECAR ../vdw_kernel.bindat .
    [ -f ../CHGCAR ] && cp ../CHGCAR .
    
    # Asegura NCORE para paralelización eficiente
    grep -q "NCORE" INCAR || echo "NCORE = $NUMCORES_PER_BAND" >> INCAR
    
    # Ejecuta VASP con optimización completa
    time mpirun --bind-to core:overload-allowed --map-by socket -np $SLURM_NTASKS --report-bindings ${VASP_DIR}/vasp_std
    vasp_exit=$?
    rm core* 2>/dev/null
    
    # Valida ejecución exitosa
    check_vasp_exit $vasp_exit "$currentdir"
    
    # Extrae información energética y de fuerzas
    free_energy=$(get_free_energy)
    energy_e0=$(get_energy_e0)
    max_force=$(get_max_force)
    opt_cycle=$(grep F OSZICAR | tail -n1 | awk '{print $1}')
    
    log_message "Ciclos NSW: $opt_cycle | E_free: $free_energy eV | E0: $energy_e0 eV | F_max: $max_force eV/Å"
    
    # Actualiza POSCAR y CHGCAR con resultados optimizados
    cp CONTCAR ../POSCAR
    [ -f CHGCAR ] && cp CHGCAR ../
    
    # Guarda backups (paralelizar en background para eficiencia)
    cp CONTCAR ../bckup/CONTCAR_$currentdir
    grep "F=" OSZICAR > ../bckup/OSZICAR_$currentdir
    
    # Comprime OUTCAR en background (no bloqueante)
    tar czf ../bckup/OUTCAR_$currentdir.tar.gz OUTCAR 2>/dev/null &
    BACKUP_PID=$!
    
    # Mueve WAVECAR (sin copiar, más eficiente)
    mv WAVECAR ../
    
    # Limpia archivos pesados de esta iteración
    clean_heavy_files
    
    # Guarda checkpoint
    save_checkpoint $count1
    
    # Espera a que termine la compresión antes de continuar
    wait $BACKUP_PID 2>/dev/null || true
    
    #==================================================================
    # CRITERIO DE CONVERGENCIA
    #==================================================================
    # La convergencia por fuerzas/energía se controla vía EDIFFG en INCAR
    # Aquí solo verificamos si VASP alcanzó el criterio o geometría estable
    converged=0
    convergence_reason=""
    
    # Criterio 1: Convergencia VASP (EDIFFG satisfecho)
    if grep -q "reached required accuracy" OUTCAR; then
        converged=1
        convergence_reason="criterio EDIFFG/EDIFF del INCAR satisfecho"
    fi
    
    # Criterio 2: Pocos ciclos NSW (geometría estable, cambios mínimos)
    if [ $opt_cycle -le $CONV_CYCLES ]; then
        converged=1
        convergence_reason="${convergence_reason:+$convergence_reason + }≤$CONV_CYCLES ciclos NSW (geometría estable)"
    fi
    
    if [ $converged -eq 1 ]; then
        log_message "✓ CONVERGENCIA ALCANZADA: $convergence_reason"
    fi
    
    if [ $converged -eq 1 ]
    then 
      vaspOK=1
      log_message "Ejecutando cálculo final de punto único..."
      
      # Cálculo de punto único sobre la geometría optimizada
      mkdir SIMUx_final 
      cp INCAR POTCAR KPOINTS WAVECAR vdw_kernel.bindat SIMUx_final
      [ -f ../CHGCAR ] && cp ../CHGCAR SIMUx_final/
      cp CONTCAR SIMUx_final/POSCAR
      cd SIMUx_final
        # Asegura NSW=0 para cálculo de punto único
        sed -i 's/NSW.*/NSW = 0  # Single point calculation/' INCAR
        
        # Asegura NCORE para paralelización
        grep -q "NCORE" INCAR || echo "NCORE = $NUMCORES_PER_BAND" >> INCAR
        
        time mpirun --bind-to core:overload-allowed --map-by socket -np $SLURM_NTASKS --report-bindings ${VASP_DIR}/vasp_std
        vasp_exit=$?
        rm core* 2>/dev/null
        
        # Extrae información final
        final_energy=$(get_free_energy)
        final_e0=$(get_energy_e0)
        final_force=$(get_max_force)
        
        if [ $vasp_exit -eq 0 ]; then
            log_message "✓ Cálculo final completo: E_free=$final_energy eV, E0=$final_e0 eV, F_max=$final_force eV/Å"
        else
            log_message "⚠ ADVERTENCIA: VASP falló en cálculo final con código $vasp_exit"
        fi
  
        # Guarda resultados finales
        cp CONTCAR ../../
        cp CONTCAR ../../bckup/CONTCAR_final
        cp OUTCAR ../../
        cp OUTCAR ../../bckup/OUTCAR_final
        cp OSZICAR ../../
        cp OSZICAR ../../bckup/OSZICAR_final
        [ -f vasprun.xml ] && cp vasprun.xml ../../
        [ -f vasprun.xml ] && cp vasprun.xml ../../bckup/vasprun.xml_final
        mv WAVECAR ../../
        [ -f CHGCAR ] && mv CHGCAR ../../
        
        # Limpia archivos pesados del cálculo final
        clean_heavy_files
      cd ../
      mv SIMUx_final ../
    fi
    
    #==================================================================
    # SEGURIDAD: Previene ciclos infinitos
    #==================================================================
    if [ $count1 -ge $MAX_ITERATIONS ]
    then 
      vaspOK=1
      log_message "⚠ ADVERTENCIA: Alcanzado límite de $MAX_ITERATIONS iteraciones. Deteniendo cálculo."
      log_message "Geometría NO convergió completamente. Última F_max: $max_force eV/Å"
      
      # Guarda estado actual para análisis
      [ -f CHGCAR ] && cp CHGCAR ../
      tar cvzf ../bckup/OUTCAR_max_iter.tar.gz OUTCAR 2>/dev/null
      cp OSZICAR ../
      
      mkdir SIMUmax_$MAX_ITERATIONS
      cp INCAR POTCAR KPOINTS POSCAR CONTCAR OSZICAR SIMUmax_$MAX_ITERATIONS
      tar cvzf SIMUmax_$MAX_ITERATIONS/OUTCAR.tar.gz OUTCAR 2>/dev/null
      mv SIMUmax_$MAX_ITERATIONS ../
    fi
  cd ../
done 

#==============================================================================
# LIMPIEZA Y ORGANIZACIÓN FINAL
#==============================================================================
log_message "Iniciando limpieza y compresión de backups"
rm -rf SIMU_*
mv SIMUx_final SIMU_final 2>/dev/null || true

# Comprime todos los backups al final (más eficiente)
log_message "Comprimiendo backups..."
tar czf bckup_all.tar.gz bckup/
log_message "Backups comprimidos en bckup_all.tar.gz"

#Para ahorrar espacio, al finalizar cada calculo de vasp en su respectivo directorio se podrían borrar los archivos que 
#no se necesitan guardar para un análisis posterior

cd $SLURM_SUBMIT_DIR
log_message "Copiando resultados finales a $SLURM_SUBMIT_DIR"
cp -r ${SCRATCH}/*  ${SLURM_SUBMIT_DIR} || exit $?
mv WAVECAR ZZ_tmp_input/
mv ZZ_tmp_input/* .
log_message "Eliminando directorio temporal $SCRATCH"
rm -rf $SCRATCH || exit $?
rmdir ZZ_tmp_input 2>/dev/null || true

log_message "========================================="
log_message "OPTIMIZACIÓN COMPLETADA EXITOSAMENTE"
log_message "========================================="


