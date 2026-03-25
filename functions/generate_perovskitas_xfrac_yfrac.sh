#!/bin/bash

# Script para generar perovskitas con composición FA_x MA_(1-x) Pb I_y Br_(1-y)
# con valores fraccionarios de x e y

# Asegúrate de que este script de Python tenga permisos de ejecución (+x)
# o invócalo con python3 directamente.
SCRIPT_PY="perovskite_poscar_creator.py"
POTCAR_DIR="$HOME/software/VASP/data"  # Directorio con los POTCAR individuales

NX=2
NY=2
NZ=2
NCELLS=$((NX * NY * NZ))     # 8
N_PB=$NCELLS                  # 8 Pb
N_TOTAL_A=$NCELLS             # 8 sitios A (FA + MA)
N_TOTAL_X=$((3 * NCELLS))     # 24 sitios X (I + Br)

# --- Función para generar POTCAR ---
generar_potcar() {
    local poscar="POSCAR"
    local potcar="POTCAR"
    
    if [[ ! -f "$poscar" ]]; then
        echo "   ✗ Error: No se encontró POSCAR"
        return 1
    fi
    
    # Leer la línea de elementos (línea 6 en formato VASP)
    local elements=$(sed -n '6p' "$poscar")
    
    echo "   Generando POTCAR con elementos: $elements"
    
    > "$potcar"
    
    for element in $elements; do
        local pot_path="$POTCAR_DIR/$element/POTCAR"
        if [[ ! -f "$pot_path" ]]; then
            echo "   ✗ Error: No se encontró $pot_path"
            return 1
        fi
        cat "$pot_path" >> "$potcar"
    done
    
    echo "   ✓ POTCAR generado exitosamente"
    return 0
}

# --- Función Helper ---
# Argumentos: x_frac, y_frac
# Donde x es la fracción de FA y y es la fracción de I
generar_caso_fraccionario() {
    local x=$1  # Fracción de FA
    local y=$2  # Fracción de I

    # Calcular número de átomos
    # FA_x MA_(1-x): x es la fracción de FA
    # I_y Br_(1-y): y es la fracción de I
    
    # Usar bc para cálculos con decimales y redondear
    N_FA=$(echo "$x * $N_TOTAL_A" | bc | awk '{print int($1+0.5)}')
    N_MA=$(echo "(1 - $x) * $N_TOTAL_A" | bc | awk '{print int($1+0.5)}')
    N_I=$(echo "$y * $N_TOTAL_X" | bc | awk '{print int($1+0.5)}')
    N_BR=$(echo "(1 - $y) * $N_TOTAL_X" | bc | awk '{print int($1+0.5)}')
    
    # Asegurar que las sumas sean exactas
    N_MA=$((N_TOTAL_A - N_FA))
    N_BR=$((N_TOTAL_X - N_I))

    # Convertir fracciones a formato de nombre (sin puntos decimales)
    x_str=$(echo "$x" | sed 's/\.//g')
    y_str=$(echo "$y" | sed 's/\.//g')
    
    # Nombre de carpeta descriptivo
    FOLDER="dir_x${x}_y${y}"

    # Evitar sobreescribir si ya existe
    if [ -d "$FOLDER" ]; then
        echo "   -> $FOLDER ya existe, saltando..."
        return
    fi

    echo "Generando: $FOLDER (FA=$N_FA, MA=$N_MA, I=$N_I, Br=$N_BR)"
    mkdir -p "$FOLDER"
    cd "$FOLDER"

    # Crear input
    cat > input_supercell_creator <<EOF
$NX
$NY
$NZ
$N_FA
$N_MA
0
$N_PB
0
$N_I
$N_BR
0
EOF

    # Ejecutar script Python
    "$SCRIPT_PY" > log.txt 2>&1
    
    if [ $? -eq 0 ]; then
        echo "   ✓ POSCAR generado correctamente"
        # Generar POTCAR
        generar_potcar
    else
        echo "   ✗ Error en la generación del POSCAR (ver log.txt)"
    fi
    
    cd ..
}

echo "=========================================="
echo " Generación de Perovskitas con x, y ∈ {0.25, 0.33, 0.66, 0.75}"
echo " Composición: FA_x MA_(1-x) Pb I_y Br_(1-y)"
echo "=========================================="
echo ""

# Arrays de fracciones a explorar
X_FRACTIONS=(0.25 0.33 0.66 0.75)
Y_FRACTIONS=(0.25 0.33 0.66 0.75)

# Recorrer todas las combinaciones
for x in "${X_FRACTIONS[@]}"; do
    for y in "${Y_FRACTIONS[@]}"; do
        generar_caso_fraccionario $x $y
    done
done

echo ""
echo "=========================================="
echo "✔ Proceso finalizado."
echo "=========================================="
echo ""
echo "Resumen de carpetas generadas:"
ls -d dir_x* 2>/dev/null | wc -l | xargs echo "Total de configuraciones:"
echo ""
echo "Para verificar las composiciones:"
echo "  for dir in dir_x*/; do echo \$dir; head -1 \$dir/POSCAR 2>/dev/null; done"
