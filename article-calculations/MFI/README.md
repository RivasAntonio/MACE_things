# MFI - Scripts de Análisis

## Estructura MFI (Zeolita tipo SiO2)

Esta carpeta contiene scripts para analizar la zeolita MFI, específicamente para comparar las energías de sus dos polimorfos: **orthorhombic** y **monoclinic**.

---

## 📁 Archivos Disponibles

### Estructuras (`structures/`)
- `CONTCAR_MFI_orthorombic.vasp` - Polimorfo orthorhombic
- `CONTCAR_MFI_monoclinic.vasp` - Polimorfo monoclinic
- `MFI_orthorombic.data`, `MFI_monoclinic.data` - Versiones LAMMPS

### Scripts Python (`python/`)

#### `mfi_phase_comparison.py` ⭐ (NUEVO - RECOMENDADO)
**Propósito:** Comparar cuantitativamente la diferencia energética entre las fases orthorhombic y monoclinic de MFI mediante minimización estructural completa.

**Qué hace:**
1. Minimiza ambas estructuras a presión 0 GPa
2. Permite relajación completa de la celda (volumen + forma)
3. Analiza parámetros de celda antes y después
4. Calcula diferencia energética precisa
5. Verifica si hay transición de fase
6. Compara volúmenes y densidades

**Uso:**
```bash
cd python/
python mfi_phase_comparison.py
```

**Outputs generados (`outputs_phase_comparison/`):**

**Estructuras optimizadas:**
- `mfi_orthorhombic_minimized.{vasp,xyz,cif}`
- `mfi_monoclinic_minimized.{vasp,xyz,cif}`

**Datos de optimización:**
- `mfi_orthorhombic_minimization.{traj,log}`
- `mfi_monoclinic_minimization.{traj,log}`

**Resumen:**
- `mfi_phase_comparison_summary.txt` - Archivo de texto con todos los resultados

---

#### `mfi_compare_energies_ase.py` (EXISTENTE)
**Propósito:** Comparación extendida con múltiples optimizadores, heads y configuraciones.

Compara:
- Dos heads: `Default` y `pt_head`
- Dos configuraciones: CUDA+CuEq vs CPU
- Dos optimizadores: BFGS vs PreconLBFGS

**Nota:** Más completo pero más lento. Útil para verificar consistencia del modelo.

---

#### `mfi_compare_energies_cuda.py` (EXISTENTE)
**Propósito:** Versión optimizada solo para CUDA+CuEq con outputs organizados.

Similar a `mfi_phase_comparison.py` pero más simple, solo BFGS.

---

## 📊 Información que proporciona `mfi_phase_comparison.py`

### Diferencia Energética
- **ΔE total** (eV)
- **ΔE por átomo** (eV/atom, meV/atom, kJ/mol)
- **Fase más estable** (menor energía)

### Parámetros Estructurales
- **Parámetros de celda:** a, b, c, α, β, γ
- **Volumen** de cada fase
- **Cambios relativos** durante la optimización
- **Densidad atómica**

### Verificación de Simetría
- Tipo de celda inicial (Ortho/Mono)
- Tipo de celda final (Ortho/Mono)
- Detección de posibles transiciones de fase

---

## ⚙️ Parámetros Configurables

```python
# Presión externa
pressure_gpa = 0.0  # GPa

# Criterio de convergencia
fmax = 0.01  # eV/Å

# Modelo
model_path = "../../zeolite-mh-finetuning.model"

# Estructuras
structures = {
    'orthorhombic': "../structures/CONTCAR_MFI_orthorombic.vasp",
    'monoclinic': "../structures/CONTCAR_MFI_monoclinic.vasp"
}
```

---

## 🎯 Interpretación de Resultados

### Diferencia Energética Típica

Para polimorfos de zeolitas:
- **ΔE < 1 meV/atom:** Fases casi degeneradas (equilibrio competitivo)
- **1-10 meV/atom:** Una fase claramente favorecida, pero otra podría formarse
- **> 10 meV/atom:** Solo una fase es estable a condiciones normales

### Ejemplo de Output Esperado

```
DIFERENCIA ENERGÉTICA (Ortho - Mono):
  ΔE = -0.023456 eV
  ΔE/atom = -0.000245 eV/atom
  ΔE/atom = -0.245 meV/atom
  ΔE/atom = -0.0236 kJ/mol

CONCLUSIÓN:
  La fase ORTHORHOMBIC es MÁS ESTABLE
  Diferencia de energía: 0.245 meV/atom
```

### Significado Físico

- **Orthorhombic más estable:** Típico a bajas presiones/temperaturas
- **Monoclinic más estable:** Puede aparecer bajo ciertas condiciones
- **Energías muy cercanas:** Transición de fase posible con T o P

---

## 🔧 Análisis Complementarios

### Cambiar presión
```python
pressure_gpa = 1.0  # Estudiar efecto de presión
```

### Verificar convergencia
```python
fmax = 0.001  # Más estricto (más lento)
```

### Usar diferentes modelos
```python
model_path = "../../zeolite-pt-head.model-mliap_lammps.pt"
```

---

## 📈 Workflow Recomendado

1. **Ejecutar `mfi_phase_comparison.py`**
   ```bash
   python mfi_phase_comparison.py
   ```
   
2. **Revisar resumen:**
   ```bash
   cat outputs_phase_comparison/mfi_phase_comparison_summary.txt
   ```

3. **Visualizar estructuras optimizadas** (con VESTA, Ovito, etc.):
   ```
   outputs_phase_comparison/mfi_orthorhombic_minimized.vasp
   outputs_phase_comparison/mfi_monoclinic_minimized.vasp
   ```

4. **Analizar trayectorias de optimización** (si es necesario):
   ```python
   from ase.io import read
   traj = read('outputs_phase_comparison/mfi_orthorhombic_minimization.traj', ':')
   energies = [atoms.get_potential_energy() for atoms in traj]
   ```

---

## 🔬 Experimentos Adicionales Sugeridos

### 1. Efecto de presión
Ejecutar con diferentes presiones para encontrar transición de fase:
```python
for P in [0.0, 0.5, 1.0, 2.0, 5.0]:  # GPa
    pressure_gpa = P
    # ejecutar minimización
```

### 2. Efecto de temperatura
Usar MD NPT para ver estabilidad dinámica:
```python
# Basado en los scripts de AFI/FAU
# Añadir NPT a diferentes T
```

### 3. Verificar con diferentes heads
Si el modelo tiene múltiples heads (como en `mfi_compare_energies_ase.py`)

### 4. Barreras de transición
Usar NEB (Nudged Elastic Band) para encontrar barrera ortho ↔ mono

---

## ⏱️ Tiempo de Ejecución

**Configuración típica:**
- Minimización orthorhombic: ~5-10 min
- Minimización monoclinic: ~5-10 min
- **Total:** ~10-20 min

**Factores que afectan:**
- Tamaño del sistema (número de átomos)
- Convergencia `fmax` (más estricto = más lento)
- GPU vs CPU
- Complejidad del modelo

---

## 🆘 Troubleshooting

**Error: CUDA not available**
→ Cambiar `device="cuda"` a `device="cpu"`

**Error: Model not found**
→ Verificar `../../zeolite-mh-finetuning.model`

**Optimización no converge**
→ Aumentar `fmax` a 0.05 o usar otro optimizador
→ Verificar estructura inicial (puede tener átomos superpuestos)

**Ambas fases convergen a la misma simetría**
→ El modelo podría favorecer una simetría específica
→ Verificar con estructuras iniciales diferentes
→ Probar optimización por pasos (solo celda, solo átomos, luego ambos)

**Diferencia energética muy grande**
→ Verificar que ambas estructuras tienen el mismo número de átomos
→ Revisar que la composición química es idéntica
→ Asegurarse que las estructuras iniciales son razonables

---

## 📝 Notas Importantes

1. **Número de átomos:** El script verifica que ambas estructuras tengan el mismo número de átomos antes de comparar

2. **Unidades:** Todas las conversiones de unidades están incluidas (eV, meV, kJ/mol)

3. **Simetría:** El script detecta automáticamente si la celda es orthorhombic (ángulos ~90°) o monoclinic

4. **Formato de salida:** Compatible con múltiples formatos (VASP, XYZ, CIF) para análisis posterior

5. **Reproducibilidad:** Todos los parámetros están documentados en el resumen

---

## 📚 Contexto Científico

### MFI (ZSM-5)
- Una de las zeolitas más importantes industrialmente
- Catálisis, separación de gases, refinado de petróleo
- Dos polimorfos conocidos: orthorhombic y monoclinic

### Transición de Fase
- Relacionada con cambios en la distribución de defectos
- Puede ocurrir con variaciones de T, P, o contenido de agua
- Relevante para propiedades catalíticas

### Estabilidad Relativa
- Importante para síntesis y procesamiento
- Afecta propiedades mecánicas y térmicas
- Puede influir en la selectividad catalítica

---

## 🎓 Para Publicación

Este script genera todos los datos necesarios para reportar:
- Diferencia energética entre polimorfos (±0.000001 eV/atom)
- Parámetros de celda optimizados (±0.000001 Å, ±0.01°)
- Volúmenes moleculares
- Método y criterios de convergencia
- Estructuras CIF para repositorios cristalográficos
