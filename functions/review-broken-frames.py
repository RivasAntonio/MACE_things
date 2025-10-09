#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from ase.io import iread
from ase.io.extxyz import XYZError

if len(sys.argv) < 2:
    print("Uso: python a.py archivo.xyz")
    sys.exit(1)

filename = sys.argv[1]

broken_frames = []

i = 0
try:
    for atoms in iread(filename, format="extxyz"):
        i += 1
        # intento de lectura ya hecho por iread
except XYZError as e:
    broken_frames.append((i+1, str(e)))

# Ahora recorremos todos los frames manualmente para capturar todos los errores
i = 0
with open(filename, 'r') as f:
    lines = f.readlines()

while i < len(lines):
    try:
        natoms = int(lines[i].strip())
        frame_lines = lines[i+2:i+2+natoms]  # +2 porque la línea de átomos y comentario
        if len(frame_lines) != natoms:
            broken_frames.append((i//(natoms+2)+1, f"Frame incompleto: tiene {len(frame_lines)} átomos, esperaba {natoms}"))
        i += natoms + 2
    except (ValueError, IndexError):
        # Cabecera inválida o fin de archivo
        break

if broken_frames:
    print("❌ Se encontraron frames rotos:")
    for frame_num, msg in broken_frames:
        print(f"Frame {frame_num}: {msg}")
else:
    print("✅ Todos los frames están correctos")

