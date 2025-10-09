#!/bin/bash

# ======================================
# Script para generar cálculos de átomos aislados
# Funcional: r2SCAN + rVV10
# Spin polarizado
# ENCUT = 1.4 * ENMAX
# ======================================

# Ruta base de los POTCAR
POTCAR_BASE=~/software/VASP/data

# Comprobamos que se han pasado elementos
if [ $# -eq 0 ]; then
  echo "Uso: $0 ELEMENTO1 ELEMENTO2 ELEMENTO3 ..."
  exit 1
fi

# Parámetros de la celda cúbica
CELL=25.0

for ELEMENT in "$@"; do
  DIRNAME="dir_${ELEMENT}"
  echo "=== Creando carpeta $DIRNAME ==="
  mkdir -p "$DIRNAME"
  cd "$DIRNAME" || exit 1

  # ================== POSCAR ==================
  cat > POSCAR <<EOF
$ELEMENT isolated atom
1.0
$CELL 0.0 0.0
0.0 $CELL 0.0
0.0 0.0 $CELL
$ELEMENT
1
Cartesian
0.0 0.0 0.0
EOF

  # ================== KPOINTS ==================
  cat > KPOINTS <<EOF
KPOINTS
0
Gamma
1 1 1
0 0 0
EOF

  # ================== POTCAR ==================
  POTCAR_FOUND=""
  for DIR in $(ls -1 "$POTCAR_BASE" | grep -E "^${ELEMENT}(_|$)"); do
    if [ -f "$POTCAR_BASE/$DIR/POTCAR" ]; then
      POTCAR_FOUND="$POTCAR_BASE/$DIR/POTCAR"
      break
    fi
  done

  if [ -n "$POTCAR_FOUND" ]; then
    cp "$POTCAR_FOUND" POTCAR
    echo "→ POTCAR copiado desde: $POTCAR_FOUND"
  else
    echo "⚠️  No se encontró POTCAR para $ELEMENT en $POTCAR_BASE"
    touch POTCAR
  fi

  # ================== INCAR ==================
  if [ -s POTCAR ]; then
    ENMAX=$(grep -m1 ENMAX POTCAR | awk '{print $3}')
    if [ -n "$ENMAX" ]; then
      ENCUT=$(awk "BEGIN {printf \"%.0f\", 1.4 * $ENMAX}")
    else
      ENCUT=520
    fi
  else
    ENCUT=520
  fi

  cat > INCAR <<EOF
# ======= Isolated atom calculation =======
SYSTEM = $ELEMENT isolated atom
ISTART = 0
ICHARG = 2
ISPIN = 2

# ======= Functional: r2SCAN + rVV10 =======
METAGGA = r2SCAN
LUSE_VDW = .TRUE.
BPARAM = 15.7
CPARAM = 0.0093

# ======= Cutoff and precision =======
ENCUT = $ENCUT
PREC = Accurate

# ======= Electronic minimization =======
EDIFF = 1E-6
ISMEAR = -5
SIGMA = 0.05
NELM = 200

# ======= Ionic settings =======
IBRION = -1
NSW = 0

# ======= Output =======
LWAVE = .TRUE.
LCHARG = .TRUE.
LREAL = .FALSE.

EOF

  echo "→ INCAR generado con ENCUT = $ENCUT eV"

  cd ..
done

echo "✅ Todo listo. Carpetas dir_ELEMENT creadas con POSCAR, KPOINTS, POTCAR e INCAR."
