# AFI - Scripts de Análisis

## Estructura AFI (Zeolita tipo SiO2)

Esta carpeta contiene scripts para analizar la zeolita AFI, específicamente para estudiar la linealidad de los ángulos en el eje ĉ.

---

## 📁 Archivos Disponibles

### Estructuras (`structures/`)
- `CONTCAR_AFI.vasp` - Estructura AFI regular
- `CONTCAR_AFI_MS_linear.vasp` - Estructura AFI con ángulos lineales forzados en Materials Studio
- `AFI.xyz`, `AFI_MS_linear.xyz` - Versiones en formato XYZ

### Scripts Python (`python/`)

#### 1. `afi_minimize_linearity.py` ⭐
**Propósito:** Minimizar ambas estructuras AFI a presión 0 y verificar si convergen a ángulos lineales (ortogonales).

**Qué hace:**
- Minimiza `CONTCAR_AFI.vasp` y `CONTCAR_AFI_MS_linear.vasp`
- Permite relajación completa de la celda
- Compara parámetros de celda antes y después
- Verifica si los ángulos α, β, γ convergen a 90° (linealidad)
- Compara energías finales

**Uso:**
```bash
cd python/
python afi_minimize_linearity.py
```

**Outputs generados (`outputs_minimization/`):**
- `AFI_minimized.{vasp,xyz,cif}` - Estructuras optimizadas
- `AFI_MS_linear_minimized.{vasp,xyz,cif}` - Estructuras optimizadas
- `*.traj` - Trayectorias de optimización
- `*.log` - Logs de optimización

**Parámetros configurables:**
- `pressure_gpa`: Presión externa (default: 0.0 GPa)
- `fmax`: Criterio de convergencia (default: 0.01 eV/Å)

---

#### 2. `afi_md_angle_histogram.py` ⭐
**Propósito:** Realizar dinámica molecular NPT y generar histogramas de la distribución de ángulos de celda.

**Qué hace:**
- Dinámica molecular NPT a temperatura controlada
- Fase de equilibración (10 ps) + fase de producción (50 ps)
- Recolecta ángulos α, β, γ durante la simulación
- Genera histogramas de distribución
- Analiza si los ángulos se mantienen lineales bajo agitación térmica

**Uso:**
```bash
cd python/
python afi_md_angle_histogram.py
```

**Outputs generados (`outputs_md_angles/`):**
- `afi_md_T300K_P0GPa.traj` - Trayectoria completa MD
- `afi_md_data_T300K.txt` - Datos numéricos (tiempo, ángulos, volumen, energía, etc.)
- `afi_angles_histogram_T300K.png` - Histogramas de α, β, γ
- `afi_angles_evolution_T300K.png` - Evolución temporal de ángulos
- `afi_thermodynamics_T300K.png` - Temperatura, volumen, energías vs tiempo

**Parámetros configurables:**
```python
temperature_K = 300.0        # Temperatura (K)
pressure_GPa = 0.0           # Presión (GPa)
timestep_fs = 0.5            # Paso de tiempo (fs)
equilibration_ps = 10.0      # Tiempo de equilibración (ps)
production_ps = 50.0         # Tiempo de producción (ps)
input_structure = "../structures/CONTCAR_AFI.vasp"  # Cambiar a AFI_MS_linear.vasp si quieres
```

---

## 🎯 Workflow Recomendado

1. **Primero:** Ejecutar `afi_minimize_linearity.py`
   - Determinar si ambas estructuras convergen a ángulos lineales
   - Comparar energías finales

2. **Segundo:** Ejecutar `afi_md_angle_histogram.py`
   - Usar la estructura minimizada como input (opcional)
   - Analizar la distribución de ángulos a temperatura finita
   - Verificar si la linealidad se mantiene durante MD

---

## 📊 Preguntas que responden estos scripts

### `afi_minimize_linearity.py`
- ¿Ambas estructuras convergen a la misma geometría?
- ¿Los ángulos se vuelven ortogonales (90°) tras la minimización?
- ¿Cuál es la diferencia energética entre las estructuras iniciales?
- ¿Cuánto cambian los parámetros de celda durante la optimización?

### `afi_md_angle_histogram.py`
- ¿Cuál es la distribución estadística de los ángulos a 300 K?
- ¿Los ángulos fluctúan alrededor de 90° o tienen otra preferencia?
- ¿Qué tan amplias son las fluctuaciones térmicas de los ángulos?
- ¿La celda se mantiene ortogonal bajo agitación térmica?

---

## ⚙️ Configuración Técnica

**Modelo MACE:** `../../zeolite-mh-finetuning.model`

**Características:**
- CuEq activado (`enable_cueq=True`) para acelerar cálculos
- Device: CUDA (GPU)
- Precisión: float32

**Optimización:**
- Algoritmo: BFGS con `UnitCellFilter` (permite relajación de celda)
- Presión: Controlable vía `scalar_pressure`

**Dinámica Molecular:**
- Ensemble: NPT (temperatura y presión constantes)
- Termostato/Barostato: Nose-Hoover
- Timestep: 0.5 fs

---

## 📝 Notas

- Los scripts están diseñados para usar CUDA (GPU). Si solo tienes CPU, cambia `device="cuda"` a `device="cpu"`
- Los tiempos de simulación son configurables según tus necesidades computacionales
- Todos los outputs se guardan en carpetas separadas para mantener organización
- Los gráficos se generan automáticamente en formato PNG de alta resolución (300 DPI)

---

## 🆘 Troubleshooting

**Error: CUDA not available**
→ Cambiar `device="cuda"` a `device="cpu"` en los scripts

**Error: Model not found**
→ Verificar que `zeolite-mh-finetuning.model` existe en `../../`

**Simulación muy lenta**
→ Reducir `production_ps` o aumentar `dump_interval`

**Memoria insuficiente**
→ Reducir `dump_interval` para guardar menos frames en la trayectoria
