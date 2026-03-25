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
    echo "       • otros      → Cualquier otro valor de IBRION"
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
    echo "    • Tipo de cálculo (MD/SP/MIN/OTROS)"
    echo "    • IBRION (valor numérico)"
    echo "    • Funcional (PBE/R2SCAN/SCAN/otros)"
    echo "    • vdW (rVV10/D3/no-vdW/other-vdW)"
    echo "    • ENCUT"
    echo "    • Número de configuraciones"
    echo "    • Composición (ej: Si64-O128)"
    echo "    • Presión (GPa)"
    echo ""
    echo -e "    Además, organiza los archivos en un directorio ${GREEN}organized_vasprun/${NC}"
    echo "    con la siguiente estructura:"
    echo "    • IBRION/Funcional/vdW/ENCUT/"
    echo "    • Nombres: composicion_N_configs_presion_kbar.xml"
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
RED='\033[0;31m'
NC='\033[0m' # No Color

###############################################################################
# VERIFICACIÓN DE DEPENDENCIAS
###############################################################################
check_dependencies() {
    local missing_deps=()
    
    # Solo verificar bc que podría no estar instalado en algunas distros mínimas
    if ! command -v bc &> /dev/null; then
        missing_deps+=("bc")
    fi
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        echo -e "${RED}ERROR: Faltan las siguientes dependencias:${NC}"
        for dep in "${missing_deps[@]}"; do
            echo "  - $dep"
        done
        echo ""
        echo "Instala con: sudo apt install bc (Debian/Ubuntu) o sudo yum install bc (RHEL/CentOS)"
        exit 1
    fi
}

# Verificar dependencias al inicio
check_dependencies

# Verificar si se solicita ayuda
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
fi

# Directorio de búsqueda
SEARCH_DIR="${1:-.}"

###############################################################################
# FUNCIÓN OPTIMIZADA: Extrae todos los parámetros en una sola lectura del archivo
###############################################################################
# Esta función lee el archivo una sola vez y extrae todos los parámetros necesarios
# Devuelve los valores separados por tabuladores para fácil parseo
extract_all_params() {
    local xml_file="$1"
    
    awk '
    BEGIN {
        ibrion = ""
        metagga = ""
        gga = ""
        encut = ""
        n_calc = 0
        pstress = ""
        tebeg = ""
        teend = ""
        ivdw_nl = ""
        ivdw = ""
        luse_vdw = ""
        bparam = ""
        cparam = ""
        has_modeling_close = 0
        composition_data = ""
    }
    
    # Detectar cierre de archivo (validación)
    /<\/modeling>/ { has_modeling_close = 1 }
    
    # IBRION
    /name="IBRION"/ && ibrion == "" {
        gsub(/.*>/, "")
        gsub(/<.*/, "")
        gsub(/[ \t]/, "")
        ibrion = $0
    }
    
    # METAGGA
    /name="METAGGA"/ && metagga == "" {
        gsub(/.*>/, "")
        gsub(/<.*/, "")
        gsub(/[ \t]/, "")
        metagga = toupper($0)
    }
    
    # GGA
    /name="GGA"/ && gga == "" {
        gsub(/.*>/, "")
        gsub(/<.*/, "")
        gsub(/[ \t]/, "")
        gga = toupper($0)
    }
    
    # ENCUT (primera ocurrencia, evitar ENCUT4O, ENCUTGW, etc.)
    /name="ENCUT"/ && encut == "" && !/ENCUT[A-Z0-9]/ {
        gsub(/.*>/, "")
        gsub(/<.*/, "")
        gsub(/[ \t]/, "")
        encut = $0
    }
    
    # Contar estructuras
    /<\/calculation>/ { n_calc++ }
    
    # PSTRESS
    /name="PSTRESS"/ && pstress == "" {
        gsub(/.*>/, "")
        gsub(/<.*/, "")
        gsub(/[ \t]/, "")
        pstress = $0
    }
    
    # TEBEG
    /name="TEBEG"/ && tebeg == "" {
        gsub(/.*>/, "")
        gsub(/<.*/, "")
        gsub(/[ \t]/, "")
        tebeg = $0
    }
    
    # TEEND
    /name="TEEND"/ && teend == "" {
        gsub(/.*>/, "")
        gsub(/<.*/, "")
        gsub(/[ \t]/, "")
        teend = $0
    }
    
    # IVDW_NL
    /name="IVDW_NL"/ && ivdw_nl == "" {
        gsub(/.*>/, "")
        gsub(/<.*/, "")
        gsub(/[ \t]/, "")
        ivdw_nl = $0
    }
    
    # IVDW (pero no IVDW_NL)
    /name="IVDW"/ && !/IVDW_NL/ && ivdw == "" {
        gsub(/.*>/, "")
        gsub(/<.*/, "")
        gsub(/[ \t]/, "")
        ivdw = $0
    }
    
    # LUSE_VDW
    /name="LUSE_VDW"/ && luse_vdw == "" {
        gsub(/.*>/, "")
        gsub(/<.*/, "")
        gsub(/[ \t]/, "")
        luse_vdw = toupper($0)
    }
    
    # BPARAM
    /name="BPARAM"/ && bparam == "" {
        gsub(/.*>/, "")
        gsub(/<.*/, "")
        gsub(/[ \t]/, "")
        bparam = $0
    }
    
    # CPARAM
    /name="CPARAM"/ && cparam == "" {
        gsub(/.*>/, "")
        gsub(/<.*/, "")
        gsub(/[ \t]/, "")
        cparam = $0
    }
    
    # Composición: buscar patrones <rc><c> N </c><c> Elemento
    /<rc><c>[ ]*[0-9]+<\/c><c>/ {
        line = $0
        gsub(/.*<rc><c>[ ]*/, "", line)
        gsub(/<\/c><c>[ ]*/, " ", line)
        gsub(/<.*/, "", line)
        split(line, parts, " ")
        if (length(parts) >= 2) {
            composition_data = composition_data parts[2] parts[1] "\n"
        }
    }
    
    END {
        # Determinar funcional
        funcional = "otros"
        if (metagga ~ /R2SCAN/) funcional = "R2SCAN"
        else if (metagga ~ /SCAN/) funcional = "SCAN"
        else if (gga ~ /PE/) funcional = "PBE"
        
        # Determinar vdW
        vdw = "other-vdW"
        # rVV10 método 1: IVDW_NL = 2
        if (ivdw_nl == "2") vdw = "rVV10"
        # rVV10 método 2: LUSE_VDW = T y BPARAM = 11.95 y CPARAM = 0.0093
        else if ((luse_vdw ~ /T|TRUE/) && bparam ~ /11\.95/ && cparam ~ /0\.0093/) vdw = "rVV10"
        # D3: IVDW = 11 o 12
        else if (ivdw == "11" || ivdw == "12") vdw = "D3"
        # no-vdW: LUSE_VDW = F y IVDW_NL = -1 y IVDW != 11 o 12
        else if ((luse_vdw ~ /F|FALSE/) && ivdw_nl == "-1" && ivdw != "11" && ivdw != "12") vdw = "no-vdW"
        # no-vdW: LUSE_VDW = F y no hay IVDW
        else if ((luse_vdw ~ /F|FALSE/) && ivdw == "") vdw = "no-vdW"
        
        # Temperatura: usar TEBEG, si es 0 o vacío usar TEEND
        temp = tebeg
        if (temp == "" || temp == "0" || temp == "0.00000000") temp = teend
        
        # Procesar composición (ordenar alfabéticamente)
        n = split(composition_data, comp_arr, "\n")
        # Ordenar y unir
        comp_sorted = ""
        for (i = 1; i <= n; i++) {
            if (comp_arr[i] != "") {
                if (comp_sorted == "") comp_sorted = comp_arr[i]
                else comp_sorted = comp_sorted "-" comp_arr[i]
            }
        }
        
        # Imprimir todos los valores separados por tabuladores
        # Formato: ibrion\tfuncional\tvdw\tencut\tn_calc\tpstress\ttemp\tcomposicion\tvalid
        printf "%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\t%d\n", ibrion, funcional, vdw, encut, n_calc, pstress, temp, comp_sorted, has_modeling_close
    }
    ' "$xml_file" 2>/dev/null
}

###############################################################################
# FUNCIÓN DE VALIDACIÓN DE ARCHIVOS XML
###############################################################################
validate_xml_file() {
    local xml_file="$1"
    local valid_flag="$2"  # 1 si tiene </modeling>, 0 si no
    
    # Si valid_flag está vacío, tratarlo como 0 (archivo truncado/inválido)
    if [ -z "$valid_flag" ]; then
        valid_flag=0
    fi
    
    # Solo verificar si tiene la etiqueta de cierre </modeling>
    # Un archivo sin esta etiqueta probablemente está truncado
    if [ "$valid_flag" -eq 0 ]; then
        echo "truncado_sin_cierre"
        return 1
    fi
    
    echo "valido"
    return 0
}


CSV_FILE="vasprun_analysis.csv"
SUMMARY_FILE="resumen_analisis.txt"

# Encabezado CSV
echo "Ruta,Tipo_Calculo,IBRION,Funcional,vdW,ENCUT,Configuraciones,Composicion,Presion_GPa,Temperatura_K" > "$CSV_FILE"

# Arrays para almacenar todas las presiones y temperaturas
ALL_PRESSURES=()
ALL_TEMPERATURES=()

# Array para archivos con advertencias (posiblemente truncados)
invalid_files=()

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
    
    # Crear estructura de directorios: IBRION/Funcional/vdW/ENCUT/
    local target_dir="${ORGANIZED_DIR}/${tipo}/${funcional}/${vdw}/${encut}"
    mkdir -p "$target_dir"
    
    # Crear nombre de archivo base: composicion_N_configs_presion_vasprun.xml
    local base_filename="${composicion}_${n_config}_configs_${presion}kbar_vasprun"
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
    
    # ═══════════════════════════════════════════════════════════════════════
    # EXTRACCIÓN OPTIMIZADA: Una sola lectura del archivo
    # ═══════════════════════════════════════════════════════════════════════
    PARAMS=$(extract_all_params "$xml_file")
    
    # Parsear los parámetros (separados por tabuladores)
    IFS=$'\t' read -r IBRION_RAW FUNCIONAL VDW ENCUT N_CONFIG PRESION_KBAR TEMPERATURA COMPOSICION VALID_FLAG <<< "$PARAMS"
    
    # ═══════════════════════════════════════════════════════════════════════
    # VALIDACIÓN DEL ARCHIVO XML
    # ═══════════════════════════════════════════════════════════════════════
    VALIDATION_RESULT=$(validate_xml_file "$xml_file" "$VALID_FLAG")
    
    if [ "$VALIDATION_RESULT" = "truncado_sin_cierre" ]; then
        # Marcamos como truncado pero lo procesamos igual, solo advertimos
        invalid_files+=("$xml_file (sin etiqueta </modeling>, posiblemente truncado)")
    fi

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

    # Convertir presión de kbar a GPa
    if [ -n "$PRESION_KBAR" ] && [[ "$PRESION_KBAR" =~ ^-?[0-9]*\.?[0-9]+$ ]]; then
        PRESION=$(echo "scale=4; $PRESION_KBAR * 0.1" | bc)
    else
        PRESION="$PRESION_KBAR"
    fi

    # Almacenar presión y temperatura para el resumen (solo valores numéricos válidos)
    if [ -n "$PRESION" ] && [[ "$PRESION" =~ ^-?[0-9]*\.?[0-9]+$ ]]; then
        ALL_PRESSURES+=("$PRESION")
    fi
    if [ -n "$TEMPERATURA" ] && [[ "$TEMPERATURA" =~ ^-?[0-9]*\.?[0-9]+$ ]]; then
        ALL_TEMPERATURES+=("$TEMPERATURA")
    fi

    # Tipo de cálculo basado en IBRION
    TIPO="$IBRION_GROUP"
    
    # Guardar en CSV sin comillas
    echo "$xml_file,$TIPO,$IBRION_RAW,$FUNCIONAL,$VDW,$ENCUT,$N_CONFIG,$COMPOSICION,$PRESION,$TEMPERATURA" >> "$CSV_FILE"
    
    # Organizar archivo en estructura de directorios
    organize_file "$xml_file" "$TIPO" "$FUNCIONAL" "$VDW" "$ENCUT" "$COMPOSICION" "$N_CONFIG" "$PRESION"
    
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

# Mostrar archivos con advertencias si hay
if [ ${#invalid_files[@]} -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}Archivos con advertencias (posiblemente truncados):${NC}"
    for item in "${invalid_files[@]}"; do
        echo "  - $item"
    done
fi

# Calcular rangos de presión y temperatura
get_min_max() {
    local arr=("$@")
    if [ ${#arr[@]} -eq 0 ]; then
        echo "N/A N/A"
        return
    fi
    local min="${arr[0]}"
    local max="${arr[0]}"
    for val in "${arr[@]}"; do
        if (( $(echo "$val < $min" | bc -l) )); then
            min="$val"
        fi
        if (( $(echo "$val > $max" | bc -l) )); then
            max="$val"
        fi
    done
    echo "$min $max"
}

# Obtener rangos
read -r PRES_MIN PRES_MAX <<< $(get_min_max "${ALL_PRESSURES[@]}")
read -r TEMP_MIN TEMP_MAX <<< $(get_min_max "${ALL_TEMPERATURES[@]}")

# Contar tipos de cálculo, funcionales y vdW desde el CSV
count_by_column() {
    local col="$1"
    tail -n +2 "$CSV_FILE" | cut -d',' -f"$col" | sort | uniq -c | sort -rn
}

# Generar archivo de resumen
{
    echo "═══════════════════════════════════════════════════════════════════════════"
    echo "                    RESUMEN DEL ANÁLISIS VASPRUN.XML"
    echo "═══════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "Fecha del análisis: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Directorio analizado: $(realpath "$SEARCH_DIR")"
    echo ""
    echo "───────────────────────────────────────────────────────────────────────────"
    echo "                         ESTADÍSTICAS GENERALES"
    echo "───────────────────────────────────────────────────────────────────────────"
    echo "Archivos procesados: $found"
    echo "Archivos saltados (no legibles): $skipped"
    echo "Archivos con advertencias: ${#invalid_files[@]}"
    echo "Total de archivos: $total"
    echo ""
    echo "───────────────────────────────────────────────────────────────────────────"
    echo "                      RANGOS DE PRESIÓN Y TEMPERATURA"
    echo "───────────────────────────────────────────────────────────────────────────"
    echo "Rango de Presiones: ${PRES_MIN} - ${PRES_MAX} GPa"
    echo "Rango de Temperaturas: ${TEMP_MIN} - ${TEMP_MAX} K"
    echo "Archivos con presión válida: ${#ALL_PRESSURES[@]}"
    echo "Archivos con temperatura válida: ${#ALL_TEMPERATURES[@]}"
    echo ""
    echo "───────────────────────────────────────────────────────────────────────────"
    echo "                       DISTRIBUCIÓN POR TIPO DE CÁLCULO"
    echo "───────────────────────────────────────────────────────────────────────────"
    count_by_column 2
    echo ""
    echo "───────────────────────────────────────────────────────────────────────────"
    echo "                         DISTRIBUCIÓN POR FUNCIONAL"
    echo "───────────────────────────────────────────────────────────────────────────"
    count_by_column 4
    echo ""
    echo "───────────────────────────────────────────────────────────────────────────"
    echo "                      DISTRIBUCIÓN POR CORRECCIÓN vdW"
    echo "───────────────────────────────────────────────────────────────────────────"
    count_by_column 5
    echo ""
    echo "───────────────────────────────────────────────────────────────────────────"
    echo "                           DISTRIBUCIÓN POR ENCUT"
    echo "───────────────────────────────────────────────────────────────────────────"
    count_by_column 6
    
    # Mostrar archivos problemáticos en el resumen si hay
    if [ ${#invalid_files[@]} -gt 0 ]; then
        echo ""
        echo "───────────────────────────────────────────────────────────────────────────"
        echo "                         ARCHIVOS CON ADVERTENCIAS"
        echo "───────────────────────────────────────────────────────────────────────────"
        echo ""
        echo "Archivos posiblemente truncados (procesados pero sin etiqueta </modeling>):"
        for item in "${invalid_files[@]}"; do
            echo "  - $item"
        done
    fi
    
    echo ""
    echo "───────────────────────────────────────────────────────────────────────────"
    echo "                            ARCHIVOS GENERADOS"
    echo "───────────────────────────────────────────────────────────────────────────"
    echo "CSV detallado: $CSV_FILE"
    echo "Archivos organizados: $ORGANIZED_DIR/"
    echo ""
    echo "═══════════════════════════════════════════════════════════════════════════"
} > "$SUMMARY_FILE"

echo "Archivo de resumen: $SUMMARY_FILE"
echo ""
cat "$SUMMARY_FILE"
