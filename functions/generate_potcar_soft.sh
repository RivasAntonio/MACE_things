#!/bin/bash

POTCAR_DIR="$HOME/software/VASP/data"  # Cambia esto si es necesario

VERBOSE=1
if [ $# -eq 0 ]; then
    BASE_DIR="."
elif [ $# -eq 1 ]; then
    if [ "$1" = "verbose" ]; then
        BASE_DIR="."
        VERBOSE=1
    else
        BASE_DIR="$1"
    fi
elif [ $# -eq 2 ]; then
    BASE_DIR="$1"
    if [ "$2" = "verbose" ]; then
        VERBOSE=1
    fi
else
    echo "Uso: $0 [directorio] [verbose]"
    exit 1
fi

if [[ ! -d "$BASE_DIR" ]]; then
    echo "Error: El directorio $BASE_DIR no existe"
    exit 1
fi

generate_potcar() {
    local dir=$1
    local poscar="$dir/POSCAR"
    local potcar="$dir/POTCAR"
    if [[ ! -f "$poscar" ]]; then
        echo "Error: No se encontró POSCAR en $dir"
        return 1
    fi
    # Leer la línea de elementos (usualmente línea 6 en formato VASP)
    local elements=$(sed -n '6p' "$poscar")
    local dir_name=$(basename "$dir")
    if [ $VERBOSE -eq 1 ]; then
        echo "Generando POTCAR para $dir_name con elementos: $elements"
    else
        echo "Generando POTCAR para $dir_name"
    fi
    > "$potcar"
    for element in $elements; do
        local pot_element="$element"
        case "$element" in
            C) pot_element="C_s" ;;
            N) pot_element="N_s" ;;
            H) pot_element="H_s" ;;
        esac

        local pot_path="$POTCAR_DIR/$pot_element/POTCAR"
        if [[ ! -f "$pot_path" ]]; then
            echo "Error: No se encontró $pot_path"
            return 1
        fi
        if [ $VERBOSE -eq 1 ]; then
            if [ "$pot_element" != "$element" ]; then
                echo "  Añadiendo $element como $pot_element desde $pot_path"
            else
                echo "  Añadiendo $element desde $pot_path"
            fi
        fi
        cat "$pot_path" >> "$potcar"
    done
    if [ $VERBOSE -eq 1 ]; then
        echo "  POTCAR generado exitosamente en $potcar"
    fi
    local vdw_src="$POTCAR_DIR/vdw_kernel.bindat"
    if [[ -f "$vdw_src" ]]; then
        cp "$vdw_src" "$dir/vdw_kernel.bindat"
        if [ $VERBOSE -eq 1 ]; then
            echo "  vdw_kernel.bindat copiado a $dir"
        fi
    else
        echo "Advertencia: No se encontró $vdw_src"
    fi
}

BASE_DIR="${BASE_DIR%/}"

TOTAL_DIRS=0
SUCCESSFUL_DIRS=0
FAILED_DIRS=0

# Recorrer recursivamente todos los POSCAR bajo el directorio base
while IFS= read -r -d '' poscar; do
    dir=$(dirname "$poscar")
    TOTAL_DIRS=$((TOTAL_DIRS+1))
    if generate_potcar "$dir"; then
        SUCCESSFUL_DIRS=$((SUCCESSFUL_DIRS+1))
    else
        FAILED_DIRS=$((FAILED_DIRS+1))
    fi
    if [ $VERBOSE -eq 1 ]; then
        echo "---"
    fi
done < <(find "$BASE_DIR" -type f -name POSCAR -print0)

if [ $TOTAL_DIRS -eq 0 ]; then
    echo "Advertencia: No se encontraron archivos POSCAR en $BASE_DIR"
    exit 0
fi

echo "Proceso completado!"
echo "Resumen:"
echo "- Directorios procesados: $TOTAL_DIRS"
echo "- Exitosos: $SUCCESSFUL_DIRS"
echo "- Fallidos: $FAILED_DIRS"
echo "- Directorio base: $BASE_DIR"

if [ $FAILED_DIRS -gt 0 ]; then
    echo "⚠️  Hubo errores en algunos directorios. Revise los mensajes anteriores."
    exit 1
else
    echo "✅ Todos los archivos POTCAR se generaron correctamente."
    exit 0
fi