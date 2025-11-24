# FAU - Scripts de Análisis

## Estructura FAU (Zeolita tipo SiO2)

Esta carpeta contiene scripts para analizar la zeolita FAU, específicamente para calcular su expansión térmica en un rango amplio de temperaturas.

---

## 📁 Archivos Disponibles

### Estructuras (`structures/`)
- `CONTCAR_FAU.vasp` - Estructura FAU en formato VASP
- `FAU.xyz` - Estructura FAU en formato XYZ

### Scripts Python (`python/`)

#### `fau_thermal_expansion.py` ⭐
**Propósito:** Calcular el coeficiente de expansión térmica de FAU mediante dinámica molecular NPT en un rango de 0-1200 K.

**Qué hace:**
1. Minimiza la estructura inicial a T=0 K y presión 0
2. Realiza barrido de temperaturas (0-1200 K, 13 puntos)
3. Para cada temperatura:
   - Equilibración NPT (20 ps)
   - Producción NPT (30 ps) para calcular promedios
   - Recolecta volumen, parámetros de celda, energía
4. Ajuste lineal V(T) = V₀ + α·V₀·T
5. Calcula coeficientes α_V (volumétrico) y α_L (lineal)
6. Genera gráficas y archivos de datos

**Uso:**
```bash
cd python/
python fau_thermal_expansion.py
```

**Outputs generados (`outputs_thermal_expansion/`):**

**Estructuras:**
- `FAU_minimized_P0GPa.vasp` - Estructura minimizada inicial
- `fau_T****K_last.vasp` - Última estructura de cada temperatura
- `fau_T****K_P0GPa.traj` - Trayectorias completas MD

**Datos:**
- `fau_expansion_data_P0GPa.txt` - Tabla con todos los datos:
  - Temperatura
  - Volumen promedio ± desviación
  - Parámetros de celda (a, b, c)
  - Energía promedio ± desviación
  - Coeficientes de expansión en el header

**Gráficas:**
- `fau_volume_vs_temp_P0GPa.png` - Volumen vs T con ajuste lineal
- `fau_cell_params_vs_temp_P0GPa.png` - Evolución de a, b, c vs T
- `fau_energy_vs_temp_P0GPa.png` - Energía potencial vs T

---

## 📊 Información que proporciona

### Coeficientes de Expansión Térmica
- **α_V** (volumétrico): Cambio relativo de volumen por grado (K⁻¹)
- **α_L** (lineal): α_V/3, aproximación para expansión lineal (K⁻¹)

### Propiedades vs Temperatura
- Volumen de la celda unidad
- Parámetros de celda (a, b, c)
- Energía potencial del sistema
- Desviaciones estándar (fluctuaciones térmicas)

### Calidad del Ajuste
- R² del ajuste lineal
- Rango de validez del comportamiento lineal

---

## ⚙️ Parámetros Configurables

```python
# Rango de temperaturas
T_min = 0           # K
T_max = 1200        # K
n_temps = 13        # Número de puntos

# Presión (¡PUEDES CAMBIARLA!)
pressure_GPa = 0.0  # GPa

# Dinámica molecular
timestep_fs = 0.25         # Paso de tiempo (fs) - como solicitaste
equilibration_ps = 20.0    # Equilibración por temperatura (ps)
production_ps = 30.0       # Producción por temperatura (ps)

# Termostato/Barostato
ttime_fs = 25.0            # Constante de tiempo termostato (fs)
pfactor_fs = 100.0         # Constante de tiempo barostato (fs)
```

---

## 🎯 Workflow del Script

```
1. Leer CONTCAR_FAU.vasp
2. Minimizar a T=0K, P=0GPa → FAU_minimized_P0GPa.vasp
3. Para cada temperatura T:
   a. Cargar estructura minimizada
   b. Inicializar velocidades a T
   c. Equilibración NPT (20 ps)
   d. Producción NPT (30 ps)
      - Recolectar datos cada ~100 pasos
      - Guardar trayectoria
   e. Calcular promedios y desviaciones
   f. Guardar última estructura
4. Ajuste lineal V vs T
5. Calcular α_V y α_L
6. Generar gráficas
7. Guardar datos en archivo .txt
```

---

## 📈 Interpretación de Resultados

### Coeficiente de Expansión Térmica Típico
Para zeolitas de SiO2, valores típicos:
- α_L ~ 5-15 × 10⁻⁶ K⁻¹ (expansión positiva)
- Algunos frameworks exhiben expansión térmica negativa (NTE)

### Gráfica Volumen vs Temperatura
- **Pendiente positiva:** Expansión térmica normal
- **Pendiente negativa:** Expansión térmica negativa (NTE)
- **Pendiente ~0:** Framework rígido, baja expansión

### Desviaciones Estándar
- Indican la magnitud de las fluctuaciones térmicas
- Aumentan con la temperatura
- Valores altos → framework flexible

---

## 🔧 Modificaciones Comunes

### Cambiar presión externa
```python
pressure_GPa = 0.5  # Por ejemplo, 0.5 GPa
```

### Más puntos de temperatura
```python
n_temps = 25  # Más fino, pero más costoso
```

### Tiempos más largos para mejor convergencia
```python
equilibration_ps = 50.0
production_ps = 100.0
```

### Rango de temperaturas diferente
```python
T_min = 100
T_max = 800
```

---

## ⏱️ Tiempo de Ejecución Estimado

**Para configuración actual:**
- Minimización inicial: ~2-5 min
- Por cada temperatura: ~10-20 min
- **Total:** ~3-5 horas (13 temperaturas)

**Para reducir tiempo:**
- Disminuir `n_temps`
- Reducir `production_ps` y `equilibration_ps`
- Aumentar `dump_interval`

---

## 🆘 Troubleshooting

**Error: CUDA not available**
→ Cambiar `device="cuda"` a `device="cpu"`

**Error: Model not found**
→ Verificar ruta de `../../zeolite-mh-finetuning.model`

**Volúmenes con mucho ruido**
→ Aumentar `production_ps` para mejores promedios

**MD no se estabiliza**
→ Aumentar `equilibration_ps`
→ Revisar `ttime_fs` y `pfactor_fs`

**Error de memoria**
→ Reducir `dump_interval` (guardar menos frames)
→ Usar chunks de temperaturas (dividir el barrido)

---

## 📝 Notas Importantes

1. **Timestep:** Se usa 0.25 fs como solicitaste (más corto que típico 0.5-1.0 fs, más estable pero más lento)

2. **Equilibración suficiente:** Los tiempos de equilibración están diseñados para que el sistema se estabilice antes de tomar datos

3. **Presión modificable:** La variable `pressure_GPa` es fácilmente modificable para estudiar el efecto de presión

4. **Archivos grandes:** Las trayectorias `.traj` pueden ocupar varios GB en total

5. **Paralelización:** Actualmente secuencial. Para paralelizar, ejecutar rangos de T en diferentes scripts

---

## 📚 Referencias

- Expansión térmica en zeolitas: Framework flexibility y breathing modes
- NPT ensemble: Control simultáneo de T y P
- Coeficiente de expansión: α_V = (1/V)(∂V/∂T)_P
