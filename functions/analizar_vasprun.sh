#!/bin/bash
###############################################################################
# analizar_vasprun.sh
#
# Script para analizar archivos vasprun.xml y generar un reporte detallado
# con información sobre tipo de cálculo, funcional, vdW, ENCUT y configuraciones.
#
# Uso:
#   ./analizar_vasprun.sh [directorio]
#   ./analizar_vasprun.sh --help
#
# Si no se proporciona directorio, busca en el directorio actual.
###############################################################################

# Función de ayuda
show_help() {
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}analizar_vasprun.sh${NC} - Analizador de archivos vasprun.xml de VASP"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${YELLOW}USO:${NC}"
    echo "    $0 [directorio]"
    echo "    $0 --help|-h"
    echo ""
    echo -e "${YELLOW}DESCRIPCIÓN:${NC}"
    echo "    Este script busca recursivamente todos los archivos vasprun.xml en un"
    echo "    directorio y genera un archivo CSV con información detallada sobre cada"
    echo "    cálculo VASP."
    echo ""
    echo -e "${YELLOW}ARGUMENTOS:${NC}"
    echo "    directorio    Directorio donde buscar archivos vasprun.xml (opcional)"
    echo "                  Por defecto: directorio actual (.)"
    echo "    --help, -h    Muestra esta ayuda"
    echo ""
    echo -e "${YELLOW}CRITERIOS DE CLASIFICACIÓN:${NC}"
    echo ""
    echo -e "    ${BLUE}1. IBRION (Tipo de cálculo):${NC}"
    echo "       • MD         → IBRION = 0  (Dinámica molecular)"
    echo "       • SP         → IBRION = -1 (Single point)"
    echo "       • MIN        → IBRION = 1, 2 o 3 (Minimización)"
    echo "       • FONONES    → IBRION = 5, 6, 7 o 8 (Cálculos de fonones)"
    echo "       • OTROS      → Cualquier otro valor de IBRION"
    echo ""
    echo -e "    ${BLUE}2. Funcional:${NC}"
    echo "       • PBE        → grep GGA | grep PE"
    echo "       • R2SCAN     → grep METAGGA | grep R2SCAN"
    echo "       • SCAN       → grep METAGGA | grep SCAN"
    echo "       • otros      → Cualquier otro funcional"
    echo ""
    echo -e "    ${BLUE}3. ENCUT:${NC}"
    echo "       Primer valor de ENCUT encontrado (evita ENCUT4O, ENCUTGW, etc.)"
    echo ""
    echo -e "    ${BLUE}4. Número de estructuras:${NC}"
    echo "       Cuenta las etiquetas </calculation> en el archivo"
    echo ""
    echo -e "    ${BLUE}5. Correcciones van der Waals:${NC}"
    echo "       • rVV10      → IVDW_NL = 2 o (LUSE_VDW = T y BPARAM = 11.95 y CPARAM = 0.0093)"
    echo "       • D3         → IVDW = 11 o 12"
    echo "       • no-vdW     → LUSE_VDW = F, IVDW_NL = -1, y IVDW ≠ 11 o 12"
    echo "       • other-vdW  → Otros tipos de correcciones"
    echo ""
    echo -e "${YELLOW}SALIDA:${NC}"
    echo -e "    Genera un archivo ${GREEN}vasprun_analysis.csv${NC} con las siguientes columnas:"
    echo "    • Ruta del archivo"
    echo "    • Tipo de cálculo (MD/SP/MIN/FONONES/OTROS)"
    echo "    • IBRION (valor numérico)"
    echo "    • Funcional (PBE/R2SCAN/SCAN/otros)"
    echo "    • vdW (rVV10/D3/no-vdW/other-vdW)"
    echo "    • ENCUT"
    echo "    • Número de configuraciones"
    echo "    • Composición (ej: Si64-O128)"
    echo "    • Presión (kbar)"
    echo "    • Temperatura (K, solo para MD)"
    echo ""
    echo -e "    Además, organiza los archivos en un directorio ${GREEN}organized_vasprun/${NC}"
    echo "    con la siguiente estructura:"
    echo "    • IBRION/Funcional/vdW/ENCUT/"
    echo "    • Nombres: composicion_N_configs_presion_kbar_temperatura_K.xml (MD)"
    echo "    • Nombres: composicion_N_configs_presion_kbar.xml (no-MD)"
    echo ""
    echo -e "${YELLOW}EJEMPLOS:${NC}"
    echo "    # Analizar el directorio actual"
    echo "    $0"
    echo ""
    echo "    # Analizar un directorio específico"
    echo "    $0 /ruta/a/calculos/vasp"
    echo ""
    echo "    # Analizar desde un servidor remoto (ejemplo)"
    echo "    $0 /mnt/beegfs/calculations/project/"
    echo ""
    echo -e "${YELLOW}NOTAS:${NC}"
    echo "    • El progreso se muestra en tiempo real durante el análisis"
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════${NC}"
    exit 0
}

# Colores para output (opcional)
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Verificar si se solicita ayuda
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
fi

# Directorio de búsqueda
SEARCH_DIR="${1:-.}"

# Función para clasificar van der Waals usando grep según criterios del usuario
classify_vdw() {
    local xml_file="$1"
    
    # Verificar rVV10 método 1: grep "IVDW_NL" | grep " 2"
    if grep -q "IVDW_NL" "$xml_file" 2>/dev/null && grep "IVDW_NL" "$xml_file" 2>/dev/null | grep -q " 2"; then
        echo "rVV10"
        return
    fi
    
    # Verificar rVV10 método 2: LUSE_VDW = T y BPARAM = 11.95 y CPARAM = 0.0093
    local luse_vdw_true=$(grep "LUSE_VDW" "$xml_file" 2>/dev/null | grep -E " T| \.TRUE\." | head -1)
    local bparam_1195=$(grep "BPARAM" "$xml_file" 2>/dev/null | grep "11.95" | head -1)
    local cparam_0093=$(grep "CPARAM" "$xml_file" 2>/dev/null | grep "0.0093" | head -1)
    
    if [ -n "$luse_vdw_true" ] && [ -n "$bparam_1195" ] && [ -n "$cparam_0093" ]; then
        echo "rVV10"
        return
    fi
    
    # Verificar D3: grep "IVDW" | grep -E " 12| 11"
    if grep -q "IVDW" "$xml_file" 2>/dev/null && grep "IVDW" "$xml_file" 2>/dev/null | grep -qE " 12| 11"; then
        echo "D3"
        return
    fi
    
    # Verificar no-vdW (cualquiera de estas condiciones es suficiente):
    # Condición 1: LUSE_VDW = F, IVDW_NL = -1, y IVDW != 12 u 11
    local luse_vdw_false=$(grep "LUSE_VDW" "$xml_file" 2>/dev/null | grep -E " F| \.FALSE\." | head -1)
    local ivdw_nl=$(grep "IVDW_NL" "$xml_file" 2>/dev/null | grep " -1" | head -1)
    local ivdw_d3=$(grep "IVDW" "$xml_file" 2>/dev/null | grep -E " 12| 11" | head -1)
    local ivdw_exists=$(grep "IVDW" "$xml_file" 2>/dev/null | head -1)
    
    # Condición 1: LUSE_VDW = F, IVDW_NL = -1, y IVDW != 12 u 11
    if [ -n "$luse_vdw_false" ] && [ -n "$ivdw_nl" ] && [ -z "$ivdw_d3" ]; then
        echo "no-vdW"
        return
    fi
    
    # Condición 2: LUSE_VDW = F y no hay IVDW
    if [ -n "$luse_vdw_false" ] && [ -z "$ivdw_exists" ]; then
        echo "no-vdW"
        return
    fi
    
    # Si no cumple ninguna condición
    echo "other-vdW"
}

# Extra helpers requested by the user
# Extract IBRION value using grep IBRION and then grepping for the numeric value (handles -1 specially)
get_ibrion() {
    local xml_file="$1"
    # Find a line containing IBRION and a numeric value (with a preceding space), prefer -1 match if present
    local val
    val=$(grep -E "IBRION" "$xml_file" 2>/dev/null | grep -E " [0-9\-]+" | head -1 | sed 's/.*>\s*\(-\?[0-9]\+\)\s*<.*/\1/')
    echo "$val"
}

# Determine functional group: PBE, R2SCAN, SCAN, or otros
get_functional_group() {
    local xml_file="$1"
    # Check METAGGA first for R2SCAN or SCAN
    if grep -q "METAGGA" "$xml_file" 2>/dev/null && grep -qi "R2SCAN" <(grep "METAGGA" "$xml_file" 2>/dev/null); then
        echo "R2SCAN"
        return
    fi
    if grep -q "METAGGA" "$xml_file" 2>/dev/null && grep -q "SCAN" <(grep "METAGGA" "$xml_file" 2>/dev/null); then
        echo "SCAN"
        return
    fi
    # Check GGA for PE -> PBE
    if grep -q "GGA" "$xml_file" 2>/dev/null && grep -q "PE" <(grep "GGA" "$xml_file" 2>/dev/null); then
        echo "PBE"
        return
    fi
    echo "otros"
}

# Get the first ENCUT occurrence (to avoid ENCUT4O, ENCUTGW, etc. take the first simple ENCUT)
get_encut_first() {
    local xml_file="$1"
    # Take first line containing ENCUT, then extract the value between > <
    local line encut
    line=$(LC_ALL=C grep "ENCUT" "$xml_file" 2>/dev/null | head -1 || true)
    if [ -n "$line" ]; then
        # Extract first numeric value and force C locale throughout
        encut=$(echo "$line" | LC_ALL=C sed 's/.*>\s*\([^<]*\)\s*<.*/\1/' | LC_ALL=C awk '{print $1}')
        # Format to 2 decimal places if encut is not empty, force C locale for dot decimal separator
        if [ -n "$encut" ]; then
            encut=$(LC_NUMERIC=C printf "%.2f" "$encut" 2>/dev/null || echo "$encut")
        fi
        echo "$encut"
    else
        echo ""
    fi
}

# Count structures using closing tag </calculation> (no need to divide by 2)
count_structures() {
    local xml_file="$1"
    local n
    n=$(grep -c "</calculation>" "$xml_file" 2>/dev/null || echo 0)
    echo "$n"
}

# Get composition from vasprun.xml (sorted alphabetically)
get_composition() {
    local xml_file="$1"
    local composition
    composition=$(grep -n "<rc><c> *[0-9]\+<\/c><c>" "$xml_file" 2>/dev/null \
        | sed -E 's/.*<c> *([0-9]+)<\/c><c> *([A-Za-z]+).*/\2\1/' \
        | sort \
        | paste -sd'-' - 2>/dev/null || echo "")
    echo "$composition"
}

# Get pressure from vasprun.xml (PSTRESS in kbar)
get_pressure() {
    local xml_file="$1"
    local pressure
    # Extract PSTRESS value, typically appears as <i name="PSTRESS">value</i>
    # Get only the first numeric value (there are two values in the line)
    pressure=$(grep "PSTRESS" "$xml_file" 2>/dev/null | head -1 | sed 's/.*>\s*\([^<]*\)\s*<.*/\1/' | awk '{print $1}')
    # Format to 2 decimal places if pressure is not empty, force C locale for dot decimal separator
    if [ -n "$pressure" ]; then
        pressure=$(LC_NUMERIC=C printf "%.2f" "$pressure" 2>/dev/null || echo "$pressure")
    fi
    echo "$pressure"
}

# Get maximum temperature from vasprun.xml (TEEND in K) for MD simulations
get_temperature() {
    local xml_file="$1"
    local temperature
    # Extract TEEND value, typically appears as <i name="TEEND">value</i>
    # Get only the first numeric value (there are two values in the line)
    temperature=$(grep "TEEND" "$xml_file" 2>/dev/null | head -1 | sed 's/.*>\s*\([^<]*\)\s*<.*/\1/' | awk '{print $1}')
    # Format to 0 decimal places (integer) if temperature is not empty, force C locale for dot decimal separator
    if [ -n "$temperature" ]; then
        temperature=$(LC_NUMERIC=C printf "%.0f" "$temperature" 2>/dev/null || echo "$temperature")
    fi
    echo "$temperature"
}


CSV_FILE="vasprun_analysis.csv"

# Encabezado CSV
echo "Ruta,Tipo_Calculo,IBRION,Funcional,vdW,ENCUT,Configuraciones,Composicion,Presion_kbar,Temperatura_K" > "$CSV_FILE"

# Directorio para la organización de archivos
ORGANIZED_DIR="organized_vasprun"

# Función para organizar archivos en árbol de directorios
organize_file() {
    local xml_file="$1"
    local tipo="$2"
    local funcional="$3"
    local vdw="$4"
    local encut="$5"
    local composicion="$6"
    local n_config="$7"
    local presion="$8"
    local temperatura="$9"
    
    # Crear estructura de directorios: IBRION/Funcional/vdW/ENCUT/
    local target_dir="${ORGANIZED_DIR}/${tipo}/${funcional}/${vdw}/${encut}"
    mkdir -p "$target_dir"
    
    # Crear nombre de archivo base: composicion_N_configs_presion_temperatura_vasprun.xml
    # Los valores ya vienen formateados de las funciones get_*
    # Solo añadir temperatura si es MD (tipo = MD) y temperatura no está vacía
    if [ "$tipo" = "MD" ] && [ -n "$temperatura" ]; then
        local base_filename="${composicion}_${n_config}_configs_${presion}kbar_${temperatura}K_vasprun"
    else
        local base_filename="${composicion}_${n_config}_configs_${presion}kbar_vasprun"
    fi
    local new_filename="${base_filename}.xml"
    local target_file="${target_dir}/${new_filename}"
    
    # Si el archivo ya existe, añadir un sufijo numérico para evitar sobrescritura
    if [ -f "$target_file" ]; then
        local counter=1
        while [ -f "${target_dir}/${base_filename}_${counter}.xml" ]; do
            ((counter++))
        done
        new_filename="${base_filename}_${counter}.xml"
        target_file="${target_dir}/${new_filename}"
    fi
    
    # Copiar archivo con el nuevo nombre
    cp "$xml_file" "$target_file"
}

# Contador
total=0
found=0
skipped=0
skipped_files=()

# Mensaje inicial
echo "Procesando archivos vasprun.xml en: $SEARCH_DIR"

# Primero contar el total de archivos
total=$(find "$SEARCH_DIR" -name "*vasprun.xml" -type f 2>/dev/null | wc -l)
echo "Total de archivos encontrados: $total"
echo ""

while IFS= read -r -d '' xml_file; do
    # Verificar que el archivo sea legible
    if [ ! -r "$xml_file" ]; then
        ((skipped++))
        skipped_files+=("$xml_file (no legible)")
        continue
    fi
    
    ((found++))
    
    # Mostrar progreso en la misma línea (con \r para sobrescribir)
    printf "\rProcesados: %d/%d (saltados: %d)" "$found" "$total" "$skipped"
    
    # Extraer parámetros usando las nuevas funciones
    IBRION_RAW=$(get_ibrion "$xml_file")

    # Determinar grupo de IBRION: 0, -1, 1-3, 5-8, otros
    if [ "$IBRION_RAW" = "0" ]; then
        IBRION_GROUP="MD"
    elif [ "$IBRION_RAW" = "-1" ]; then
        IBRION_GROUP="SP"
    elif [ "$IBRION_RAW" = "1" ] || [ "$IBRION_RAW" = "2" ] || [ "$IBRION_RAW" = "3" ]; then
        IBRION_GROUP="MIN"
    elif [ "$IBRION_RAW" = "5" ] || [ "$IBRION_RAW" = "6" ] || [ "$IBRION_RAW" = "7" ] || [ "$IBRION_RAW" = "8" ]; then
        IBRION_GROUP="FONONES"
    else
        IBRION_GROUP="OTROS"
    fi

    # Funcional, ENCUT y conteo de estructuras
    FUNCIONAL=$(get_functional_group "$xml_file")
    ENCUT=$(get_encut_first "$xml_file")
    N_CONFIG=$(count_structures "$xml_file")
    COMPOSICION=$(get_composition "$xml_file")
    PRESION=$(get_pressure "$xml_file")
    TEMPERATURA=$(get_temperature "$xml_file")

    # Clasificar vdW (ahora recibe el archivo xml directamente)
    VDW=$(classify_vdw "$xml_file")
    
    # Guardar en CSV sin comillas
    echo "$xml_file,$IBRION_GROUP,$IBRION_RAW,$FUNCIONAL,$VDW,$ENCUT,$N_CONFIG,$COMPOSICION,$PRESION,$TEMPERATURA" >> "$CSV_FILE"
    
    # Organizar archivo en estructura de directorios
    organize_file "$xml_file" "$IBRION_GROUP" "$FUNCIONAL" "$VDW" "$ENCUT" "$COMPOSICION" "$N_CONFIG" "$PRESION" "$TEMPERATURA"
    
done < <(find "$SEARCH_DIR" -name "*vasprun.xml" -type f -print0 2>/dev/null)

# Salto de línea después del progreso
echo ""
echo ""

# Resumen
echo "=========================================="
echo "Resumen del análisis"
echo "=========================================="
echo "Archivos procesados: $found"
echo "Archivos saltados: $skipped"
echo "Total: $total"
echo "Archivo CSV: $CSV_FILE"
echo "Archivos organizados en: $ORGANIZED_DIR"
echo "=========================================="

# Mostrar archivos saltados si hay
if [ $skipped -gt 0 ]; then
    echo ""
    echo "Archivos saltados:"
    for item in "${skipped_files[@]}"; do
        echo "  - $item"
    done
fi

# Generar archivo de resumen detallado
SUMMARY_FILE="vasprun_analysis_summary.txt"
echo "Generando resumen detallado en: $SUMMARY_FILE"

cat > "$SUMMARY_FILE" << EOF
═══════════════════════════════════════════════════════════════════════════
                    RESUMEN DEL ANÁLISIS DE VASPRUN.XML
═══════════════════════════════════════════════════════════════════════════

Fecha y hora del análisis: $(date)
Directorio analizado: $SEARCH_DIR

───────────────────────────────────────────────────────────────────────────
1. ESTADÍSTICAS GENERALES
───────────────────────────────────────────────────────────────────────────

Total de archivos encontrados:  $total
Archivos procesados exitosamente: $found
Archivos saltados:               $skipped

───────────────────────────────────────────────────────────────────────────
2. DISTRIBUCIÓN POR TIPO DE CÁLCULO (IBRION)
───────────────────────────────────────────────────────────────────────────

EOF

# Contar por tipo de cálculo
if [ -f "$CSV_FILE" ]; then
    echo "Dinámica Molecular (MD):     $(tail -n +2 "$CSV_FILE" | cut -d',' -f2 | grep -c '^MD$')" >> "$SUMMARY_FILE"
    echo "Single Point (SP):           $(tail -n +2 "$CSV_FILE" | cut -d',' -f2 | grep -c '^SP$')" >> "$SUMMARY_FILE"
    echo "Minimización (MIN):          $(tail -n +2 "$CSV_FILE" | cut -d',' -f2 | grep -c '^MIN$')" >> "$SUMMARY_FILE"
    echo "Fonones:                     $(tail -n +2 "$CSV_FILE" | cut -d',' -f2 | grep -c '^FONONES$')" >> "$SUMMARY_FILE"
    echo "Otros:                       $(tail -n +2 "$CSV_FILE" | cut -d',' -f2 | grep -c '^OTROS$')" >> "$SUMMARY_FILE"
    
    cat >> "$SUMMARY_FILE" << EOF

───────────────────────────────────────────────────────────────────────────
3. DISTRIBUCIÓN POR FUNCIONAL
───────────────────────────────────────────────────────────────────────────

EOF
    
    echo "PBE:                         $(tail -n +2 "$CSV_FILE" | cut -d',' -f4 | grep -c '^PBE$')" >> "$SUMMARY_FILE"
    echo "SCAN:                        $(tail -n +2 "$CSV_FILE" | cut -d',' -f4 | grep -c '^SCAN$')" >> "$SUMMARY_FILE"
    echo "R2SCAN:                      $(tail -n +2 "$CSV_FILE" | cut -d',' -f4 | grep -c '^R2SCAN$')" >> "$SUMMARY_FILE"
    echo "Otros:                       $(tail -n +2 "$CSV_FILE" | cut -d',' -f4 | grep -c '^otros$')" >> "$SUMMARY_FILE"
    
    cat >> "$SUMMARY_FILE" << EOF

───────────────────────────────────────────────────────────────────────────
4. DISTRIBUCIÓN POR CORRECCIONES VAN DER WAALS
───────────────────────────────────────────────────────────────────────────

EOF
    
    echo "rVV10:                       $(tail -n +2 "$CSV_FILE" | cut -d',' -f5 | grep -c '^rVV10$')" >> "$SUMMARY_FILE"
    echo "D3:                          $(tail -n +2 "$CSV_FILE" | cut -d',' -f5 | grep -c '^D3$')" >> "$SUMMARY_FILE"
    echo "Sin vdW (no-vdW):            $(tail -n +2 "$CSV_FILE" | cut -d',' -f5 | grep -c '^no-vdW$')" >> "$SUMMARY_FILE"
    echo "Otros vdW:                   $(tail -n +2 "$CSV_FILE" | cut -d',' -f5 | grep -c '^other-vdW$')" >> "$SUMMARY_FILE"
    
    cat >> "$SUMMARY_FILE" << EOF

───────────────────────────────────────────────────────────────────────────
5. VALORES DE ENCUT ENCONTRADOS
───────────────────────────────────────────────────────────────────────────

EOF
    
    tail -n +2 "$CSV_FILE" | cut -d',' -f6 | sort -n | uniq -c | sort -rn | awk '{printf "%-10s eV:  %d archivos\n", $2, $1}' >> "$SUMMARY_FILE"
    
    cat >> "$SUMMARY_FILE" << EOF

───────────────────────────────────────────────────────────────────────────
6. ESTADÍSTICAS DE CONFIGURACIONES
───────────────────────────────────────────────────────────────────────────

EOF
    
    total_configs=$(tail -n +2 "$CSV_FILE" | cut -d',' -f7 | awk '{sum+=$1} END {print sum}')
    max_configs=$(tail -n +2 "$CSV_FILE" | cut -d',' -f7 | sort -n | tail -1)
    min_configs=$(tail -n +2 "$CSV_FILE" | cut -d',' -f7 | sort -n | head -1)
    avg_configs=$(tail -n +2 "$CSV_FILE" | cut -d',' -f7 | awk '{sum+=$1; count++} END {if(count>0) printf "%.2f", sum/count; else print 0}')
    
    echo "Total de configuraciones:    $total_configs" >> "$SUMMARY_FILE"
    echo "Promedio por archivo:        $avg_configs" >> "$SUMMARY_FILE"
    echo "Máximo en un archivo:        $max_configs" >> "$SUMMARY_FILE"
    echo "Mínimo en un archivo:        $min_configs" >> "$SUMMARY_FILE"
    
    cat >> "$SUMMARY_FILE" << EOF

───────────────────────────────────────────────────────────────────────────
7. COMPOSICIONES ENCONTRADAS
───────────────────────────────────────────────────────────────────────────

EOF
    
    tail -n +2 "$CSV_FILE" | cut -d',' -f8 | sort | uniq -c | sort -rn | head -20 | awk '{printf "%-30s %d archivos\n", $2, $1}' >> "$SUMMARY_FILE"
    
    total_compositions=$(tail -n +2 "$CSV_FILE" | cut -d',' -f8 | sort | uniq | wc -l)
    if [ $total_compositions -gt 20 ]; then
        echo "" >> "$SUMMARY_FILE"
        echo "(Mostrando las 20 composiciones más frecuentes de $total_compositions totales)" >> "$SUMMARY_FILE"
    fi
fi

cat >> "$SUMMARY_FILE" << EOF

───────────────────────────────────────────────────────────────────────────
8. ARCHIVOS GENERADOS
───────────────────────────────────────────────────────────────────────────

Archivo CSV:                 $CSV_FILE
Directorio organizado:       $ORGANIZED_DIR/
Archivo de resumen:          $SUMMARY_FILE

═══════════════════════════════════════════════════════════════════════════
                           FIN DEL RESUMEN
═══════════════════════════════════════════════════════════════════════════
EOF

echo ""
echo "✓ Resumen detallado generado: $SUMMARY_FILE"
