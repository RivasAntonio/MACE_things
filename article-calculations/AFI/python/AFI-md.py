#!/usr/bin/env python3
"""
Script para realizar dinámica molecular de AFI y generar histograma de ángulos
Equilibración: NVT (Langevin)
Producción: MTKNPT (Martyna-Tobias-Klein NPT)

Autor: Optimized version
Fecha: 2025
"""
import sys
import time
from pathlib import Path
from typing import Optional, Tuple
import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from ase import units
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import MTKNPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from mace.calculators import MACECalculator

# Suprimir warnings no críticos
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# CONSTANTES
# ============================================================================
FS_PER_PS = 1000.0  # Femtosegundos por picosegundo
KELVIN_TO_EV = 8.617333262e-5  # Factor de conversión K -> eV
ATM_TO_EV_A3 = 6.24150907e-7  # Presión: 1 atm en eV/Å³


# ============================================================================
# PARÁMETROS DE SIMULACIÓN
# ============================================================================

class SimulationConfig:
    """Configuración centralizada de la simulación"""
    
    # Archivos de entrada
    model_path: str = "../../zeolite-mh-finetuning.model"
    input_structure: str = "../structures/CONTCAR_AFI_MS_linear.vasp"
    
    # Parámetros termodinámicos
    temperature_K: float = 300.0
    pressure_atm: float = 0.0  # Presión en atmósferas (0 = vacío)
    
    # Parámetros temporales
    timestep_fs: float = 0.4  # Paso de tiempo en femtosegundos
    equilibration_ps: float = 20.0  # Tiempo de equilibración NVT
    production_ps: float = 200.0  # Tiempo de producción NPT
    dump_interval: int = 100  # Guardar cada N pasos
    
    # Parámetros del termostato y barostato (en unidades ASE)
    friction_langevin: float = 0.002  # 1/fs - Fricción para equilibración
    ttime_fs: float = 100.0  # Constante de tiempo del termostato (fs)
    ptime_fs: float = 2000.0  # Constante de tiempo del barostato (fs)
    
    # Configuración computacional
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    enable_cueq: bool = True
    dtype: str = "float64"
    
    # Control de outputs
    output_dir: str = "outputs_md_angles_initial_linear"
    generate_plots: bool = True
    plot_dpi: int = 300
    
    # Validación de equilibración
    check_equilibration: bool = True
    equilibration_check_window: int = 100  # últimos N frames para verificar
    equilibration_temp_tolerance: float = 10.0  # K
    
    @property
    def pressure_eV_A3(self) -> float:
        """Convierte presión de atm a eV/Å³"""
        return self.pressure_atm * ATM_TO_EV_A3
    
    @property
    def timestep(self) -> float:
        """Timestep en unidades ASE"""
        return self.timestep_fs * units.fs
    
    @property
    def ttime(self) -> float:
        """Tiempo del termostato en unidades ASE"""
        return self.ttime_fs * units.fs
    
    @property
    def ptime(self) -> float:
        """Tiempo del barostato en unidades ASE"""
        return self.ptime_fs * units.fs
    
    @property
    def friction(self) -> float:
        """Fricción de Langevin en unidades ASE"""
        return self.friction_langevin / units.fs
    
    def validate(self) -> Tuple[bool, str]:
        """Valida la configuración"""
        if self.timestep_fs <= 0:
            return False, "El timestep debe ser positivo"
        if self.timestep_fs > 2.0:
            return False, "Timestep muy grande (>2 fs), puede causar inestabilidad"
        if self.temperature_K <= 0:
            return False, "La temperatura debe ser positiva"
        if self.equilibration_ps <= 0 or self.production_ps <= 0:
            return False, "Los tiempos de simulación deben ser positivos"
        if self.ttime_fs < 50 or self.ttime_fs > 1000:
            return False, "ttime fuera del rango recomendado (50-1000 fs)"
        if self.ptime_fs < 500 or self.ptime_fs > 10000:
            return False, "ptime fuera del rango recomendado (500-10000 fs)"
        return True, "OK"


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def print_header(title: str, char: str = "=", width: int = 70) -> None:
    """Imprime un encabezado formateado"""
    print(char * width)
    print(f" {title}")
    print(char * width)


def print_section(title: str, width: int = 70) -> None:
    """Imprime un título de sección"""
    print(f"\n{title}")
    print("-" * width)


def estimate_trajectory_size(n_atoms: int, n_frames: int) -> float:
    """Estima el tamaño del archivo de trayectoria en MB"""
    # Estimación aproximada: ~50 bytes por átomo por frame
    return (n_atoms * n_frames * 50) / (1024 ** 2)


def check_equilibration(traj_file: Path, window: int = 100, 
                       temp_tolerance: float = 10.0) -> Tuple[bool, dict]:
    """
    Verifica si el sistema está equilibrado analizando la trayectoria
    
    Returns:
        (is_equilibrated, stats_dict)
    """
    try:
        traj = Trajectory(str(traj_file))
        if len(traj) < window:
            return False, {"error": "Trayectoria muy corta para análisis"}
        
        # Analizar últimos 'window' frames
        temps = np.array([atoms.get_temperature() for atoms in traj[-window:]])
        vols = np.array([atoms.get_volume() for atoms in traj[-window:]])
        
        temp_std = np.std(temps)
        vol_std = np.std(vols)
        
        is_equilibrated = temp_std < temp_tolerance
        
        stats = {
            "temp_mean": np.mean(temps),
            "temp_std": temp_std,
            "vol_mean": np.mean(vols),
            "vol_std": vol_std,
            "is_equilibrated": is_equilibrated
        }
        
        return is_equilibrated, stats
        
    except Exception as e:
        return False, {"error": str(e)}


def save_data_efficiently(output_dir: Path, config: SimulationConfig,
                         times: np.ndarray, data_dict: dict) -> Path:
    """Guarda datos en formato comprimido npz"""
    data_file = output_dir / f"afi_md_data_T{int(config.temperature_K)}K.npz"
    np.savez_compressed(
        data_file,
        times=times,
        **data_dict
    )
    return data_file


# ============================================================================
# CLASE PRINCIPAL DE SIMULACIÓN
# ============================================================================

class AFIMDSimulation:
    """Clase para gestionar la simulación de dinámica molecular de AFI"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.atoms = None
        self.calc = None
        self.data = {}
        
    def setup(self) -> bool:
        """Configura el sistema inicial"""
        print_header("DINÁMICA MOLECULAR - ANÁLISIS DE ÁNGULOS AFI")
        
        # Validar configuración
        is_valid, msg = self.config.validate()
        if not is_valid:
            print(f"\n❌ Error en configuración: {msg}")
            return False
        
        # Información de configuración
        print(f"\n📁 Directorio de salida: {self.output_dir}")
        print(f"\n📂 Archivos de entrada:")
        print(f"  Estructura: {self.config.input_structure}")
        print(f"  Modelo MACE: {self.config.model_path}")
        
        print(f"\n⚙️  Parámetros MD:")
        print(f"  Temperatura: {self.config.temperature_K} K")
        print(f"  Presión: {self.config.pressure_atm} atm "
              f"({self.config.pressure_eV_A3:.2e} eV/Å³)")
        print(f"  Timestep: {self.config.timestep_fs} fs")
        print(f"  Equilibración (NVT): {self.config.equilibration_ps} ps")
        print(f"  Producción (MTKNPT): {self.config.production_ps} ps")
        print(f"  Tiempo total: {(self.config.equilibration_ps + self.config.production_ps):.2f} ps")
        print(f"  Intervalo de guardado: cada {self.config.dump_interval} pasos "
              f"(~{self.config.dump_interval * self.config.timestep_fs:.1f} fs)")
        
        print(f"\n⚙️  Parámetros termostato/barostato:")
        print(f"  Fricción (Langevin): {self.config.friction_langevin:.4f} fs⁻¹")
        print(f"  ttime (MTKNPT): {self.config.ttime_fs} fs")
        print(f"  ptime (MTKNPT): {self.config.ptime_fs} fs")
        
        print(f"\n💻 Configuración computacional:")
        print(f"  CuEq: {'Activado' if self.config.enable_cueq else 'Desactivado'}")
        print(f"  Device: {self.config.device}")
        print(f"  Dtype: {self.config.dtype}")
        
        print("=" * 70)
        
        # Leer estructura
        try:
            self.atoms = read(self.config.input_structure)
        except FileNotFoundError:
            print(f"\n❌ Error: No se encontró el archivo {self.config.input_structure}")
            return False
        except Exception as e:
            print(f"\n❌ Error al leer estructura: {e}")
            return False
        
        # Inicializar calculador MACE
        try:
            self.calc = MACECalculator(
                model_paths=self.config.model_path,
                device=self.config.device,
                default_dtype=self.config.dtype,
                enable_cueq=self.config.enable_cueq
            )
            self.atoms.calc = self.calc
        except Exception as e:
            print(f"\n❌ Error al inicializar calculador MACE: {e}")
            return False
        
        # Información inicial del sistema
        print("\n📊 Estructura inicial:")
        cell_params = self.atoms.cell.cellpar()
        print(f"  Átomos: {len(self.atoms)}")
        print(f"  Fórmula: {self.atoms.get_chemical_formula()}")
        print(f"  Celda: a={cell_params[0]:.3f}, b={cell_params[1]:.3f}, "
              f"c={cell_params[2]:.3f} Å")
        print(f"  Ángulos: α={cell_params[3]:.2f}°, β={cell_params[4]:.2f}°, "
              f"γ={cell_params[5]:.2f}°")
        print(f"  Volumen: {self.atoms.get_volume():.2f} Å³")
        
        try:
            E_initial = self.atoms.get_potential_energy()
            print(f"  Energía inicial: {E_initial:.4f} eV")
            print(f"  Energía por átomo: {E_initial/len(self.atoms):.4f} eV/atom")
        except Exception as e:
            print(f"  ⚠️  No se pudo calcular energía inicial: {e}")
        
        # Advertencia sobre tamaño de archivo
        n_frames_prod = int(self.config.production_ps * FS_PER_PS / 
                           (self.config.timestep_fs * self.config.dump_interval))
        estimated_size = estimate_trajectory_size(len(self.atoms), n_frames_prod)
        if estimated_size > 500:
            print(f"\n⚠️  Advertencia: El archivo de trayectoria será ~{estimated_size:.0f} MB")
        
        return True
    
    def run_equilibration(self) -> bool:
        """Ejecuta la fase de equilibración NVT"""
        print_section("🔄 FASE 1: EQUILIBRACIÓN NVT")
        
        # Inicializar velocidades
        MaxwellBoltzmannDistribution(self.atoms, 
                                     temperature_K=self.config.temperature_K)
        
        # Archivos de salida
        equi_log = self.output_dir / f"equilibration_T{int(self.config.temperature_K)}K.log"
        equi_traj = self.output_dir / f"equilibration_T{int(self.config.temperature_K)}K.traj"
        
        # Crear dinámica NVT (Langevin)
        dyn = Langevin(
            self.atoms,
            timestep=self.config.timestep,
            temperature_K=self.config.temperature_K,
            friction=self.config.friction,
            logfile=str(equi_log),
            loginterval=self.config.dump_interval,
            trajectory=str(equi_traj)
        )
        
        # Calcular pasos
        n_steps = int(self.config.equilibration_ps * FS_PER_PS / 
                     self.config.timestep_fs)
        
        print(f"Equilibrando por {n_steps} pasos ({self.config.equilibration_ps} ps)...")
        print(f"Progreso: ", end='', flush=True)
        
        # Ejecutar con barra de progreso
        progress_interval = max(1, n_steps // 20)
        
        def print_progress():
            if dyn.nsteps % progress_interval == 0:
                progress = (dyn.nsteps / n_steps) * 100
                print(f"{progress:.0f}% ", end='', flush=True)
        
        dyn.attach(print_progress, interval=progress_interval)
        
        start_time = time.time()
        try:
            dyn.run(n_steps)
            elapsed = time.time() - start_time
        except Exception as e:
            print(f"\n❌ Error durante equilibración: {e}")
            return False
        
        print(f"\n✓ Equilibración completada en {elapsed:.1f} s")
        print(f"💾 Log: {equi_log.name}")
        print(f"💾 Trayectoria: {equi_traj.name}")
        
        # Verificar equilibración si está habilitado
        if self.config.check_equilibration:
            print("\n📊 Verificando equilibración...")
            is_eq, stats = check_equilibration(
                equi_traj,
                window=self.config.equilibration_check_window,
                temp_tolerance=self.config.equilibration_temp_tolerance
            )
            
            if "error" in stats:
                print(f"⚠️  No se pudo verificar: {stats['error']}")
            else:
                print(f"  Temperatura: {stats['temp_mean']:.2f} ± {stats['temp_std']:.2f} K")
                print(f"  Volumen: {stats['vol_mean']:.2f} ± {stats['vol_std']:.2f} Å³")
                
                if is_eq:
                    print("  ✓ Sistema equilibrado")
                else:
                    print(f"  ⚠️  Sistema posiblemente no equilibrado "
                          f"(σ_T = {stats['temp_std']:.2f} K)")
        
        return True
    
    def run_production(self) -> bool:
        """Ejecuta la fase de producción NPT"""
        print_section("🔄 FASE 2: PRODUCCIÓN MTKNPT")
        
        # Calcular pasos y frames
        n_steps = int(self.config.production_ps * FS_PER_PS / 
                     self.config.timestep_fs)
        n_samples = n_steps // self.config.dump_interval + 1
        
        print(f"Simulación de producción: {n_steps} pasos ({self.config.production_ps} ps)")
        print(f"Frames esperados: {n_samples}")
        
        # Preallocar arrays para datos
        self.data = {
            'times': np.zeros(n_samples),
            'angles_alpha': np.zeros(n_samples),
            'angles_beta': np.zeros(n_samples),
            'angles_gamma': np.zeros(n_samples),
            'volumes': np.zeros(n_samples),
            'temperatures': np.zeros(n_samples),
            'energies_pot': np.zeros(n_samples),
            'energies_kin': np.zeros(n_samples),
            'pressures': np.zeros(n_samples)
        }
        self.data_counter = 0
        
        # Archivos de salida
        prod_log = self.output_dir / f"production_T{int(self.config.temperature_K)}K_P{self.config.pressure_atm}atm.log"
        prod_traj = self.output_dir / f"production_T{int(self.config.temperature_K)}K_P{self.config.pressure_atm}atm.traj"
        
        # Abrir trayectoria
        traj = Trajectory(str(prod_traj), 'w', self.atoms)
        
        # Crear dinámica MTKNPT
        try:
            dyn = MTKNPT(
                self.atoms,
                timestep=self.config.timestep,
                temperature_K=self.config.temperature_K,
                pressure_au=self.config.pressure_eV_A3,
                tdamp=self.config.ttime,
                pdamp=self.config.ptime,  # Usar ptime del config
                logfile=str(prod_log),
                loginterval=self.config.dump_interval
            )
            # Nota: ASE puede usar 'pfactor' o calcular internamente desde 'ptime'
            # Ajustar según versión de ASE
        except TypeError:
            # Intentar con parámetros alternativos si hay problemas de versión
            dyn = MTKNPT(
                self.atoms,
                timestep=self.config.timestep,
                temperature_K=self.config.temperature_K,
                pressure_au=self.config.pressure_eV_A3,
                ttime=self.config.ttime,
                pfactor=(self.config.ptime ** 2) * self.atoms.get_masses().sum(),
                logfile=str(prod_log),
                loginterval=self.config.dump_interval
            )
        
        # Función para recolectar datos
        def collect_data():
            """Recolecta datos durante la simulación"""
            idx = self.data_counter
            
            # Parámetros de celda
            cell_params = self.atoms.cell.cellpar()
            self.data['angles_alpha'][idx] = cell_params[3]
            self.data['angles_beta'][idx] = cell_params[4]
            self.data['angles_gamma'][idx] = cell_params[5]
            
            # Propiedades termodinámicas
            self.data['volumes'][idx] = self.atoms.get_volume()
            self.data['temperatures'][idx] = self.atoms.get_temperature()
            self.data['energies_pot'][idx] = self.atoms.get_potential_energy()
            self.data['energies_kin'][idx] = self.atoms.get_kinetic_energy()
            
            # Calcular presión (si está disponible)
            try:
                stress = self.atoms.get_stress(voigt=False)
                pressure_eV_A3 = -np.trace(stress) / 3.0
                self.data['pressures'][idx] = pressure_eV_A3 / ATM_TO_EV_A3  # convertir a atm
            except:
                self.data['pressures'][idx] = np.nan
            
            # Tiempo
            self.data['times'][idx] = dyn.nsteps * self.config.timestep_fs / FS_PER_PS
            
            self.data_counter += 1
            
            # Guardar frame en trayectoria
            traj.write()
        
        # Barra de progreso
        progress_interval = max(1, n_steps // 20)
        
        def print_progress():
            if dyn.nsteps % progress_interval == 0:
                progress = (dyn.nsteps / n_steps) * 100
                print(f"\rProgreso: {progress:.0f}%", end='', flush=True)
        
        # Adjuntar callbacks
        dyn.attach(collect_data, interval=self.config.dump_interval)
        dyn.attach(print_progress, interval=progress_interval)
        
        # Ejecutar producción
        print("\nProgreso: ", end='', flush=True)
        start_time = time.time()
        
        try:
            dyn.run(n_steps)
            elapsed = time.time() - start_time
        except Exception as e:
            print(f"\n❌ Error durante producción: {e}")
            traj.close()
            return False
        
        traj.close()
        
        print(f"\n✓ Producción completada en {elapsed:.1f} s")
        print(f"💾 Log: {prod_log.name}")
        print(f"💾 Trayectoria: {prod_traj.name}")
        
        # Recortar arrays al tamaño real
        for key in self.data:
            self.data[key] = self.data[key][:self.data_counter]
        
        return True
    
    def analyze_and_save(self) -> None:
        """Analiza resultados y guarda datos"""
        print_section("📊 ANÁLISIS DE RESULTADOS")
        
        # Estadísticas de ángulos
        print("\n🔷 Estadísticas de ángulos de celda:")
        for angle, name in [('angles_alpha', 'α'), ('angles_beta', 'β'), 
                           ('angles_gamma', 'γ')]:
            data = self.data[angle]
            print(f"  {name}: {np.mean(data):.4f} ± {np.std(data):.4f}° "
                  f" [{np.min(data):.4f}, {np.max(data):.4f}]")
        
        # Estadísticas termodinámicas
        print("\n🌡️  Estadísticas termodinámicas:")
        print(f"  Temperatura: {np.mean(self.data['temperatures']):.2f} ± "
              f"{np.std(self.data['temperatures']):.2f} K "
              f"(target: {self.config.temperature_K} K)")
        print(f"  Volumen: {np.mean(self.data['volumes']):.2f} ± "
              f"{np.std(self.data['volumes']):.2f} Å³")
        print(f"  E_pot: {np.mean(self.data['energies_pot']):.4f} ± "
              f"{np.std(self.data['energies_pot']):.4f} eV")
        print(f"  E_kin: {np.mean(self.data['energies_kin']):.4f} ± "
              f"{np.std(self.data['energies_kin']):.4f} eV")
        
        if not np.all(np.isnan(self.data['pressures'])):
            print(f"  Presión: {np.nanmean(self.data['pressures']):.4f} ± "
                  f"{np.nanstd(self.data['pressures']):.4f} atm "
                  f"(target: {self.config.pressure_atm} atm)")
        
        # Guardar datos en formato npz comprimido
        data_file = save_data_efficiently(
            self.output_dir,
            self.config,
            self.data['times'],
            self.data
        )
        print(f"\n💾 Datos guardados (comprimidos): {data_file.name}")
        
        # También guardar en formato texto para compatibilidad
        txt_file = self.output_dir / f"afi_md_data_T{int(self.config.temperature_K)}K.txt"
        header = ("Time(ps) Alpha(deg) Beta(deg) Gamma(deg) Volume(A^3) "
                 "Temp(K) Epot(eV) Ekin(eV) Pressure(atm)")
        data_array = np.column_stack([
            self.data['times'],
            self.data['angles_alpha'],
            self.data['angles_beta'],
            self.data['angles_gamma'],
            self.data['volumes'],
            self.data['temperatures'],
            self.data['energies_pot'],
            self.data['energies_kin'],
            self.data['pressures']
        ])
        np.savetxt(txt_file, data_array, header=header, fmt='%.6f')
        print(f"💾 Datos guardados (texto): {txt_file.name}")
    
    def generate_plots(self) -> None:
        """Genera gráficas de análisis"""
        if not self.config.generate_plots:
            print("\n📊 Generación de gráficas desactivada")
            return
        
        print_section("📈 GENERANDO GRÁFICAS")
        
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Figura 1: Histogramas de ángulos
        self._plot_angle_histograms()
        
        # Figura 2: Evolución temporal de ángulos
        self._plot_angle_evolution()
        
        # Figura 3: Propiedades termodinámicas
        self._plot_thermodynamics()
        
        print("✓ Gráficas generadas")
    
    def _plot_angle_histograms(self) -> None:
        """Genera histogramas de ángulos"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(
            f'Histogramas de Ángulos - AFI MD NPT\n'
            f'T={self.config.temperature_K}K, P={self.config.pressure_atm}atm',
            fontsize=14, fontweight='bold'
        )
        
        angles_data = [
            (self.data['angles_alpha'], 'α (alpha)', 'red'),
            (self.data['angles_beta'], 'β (beta)', 'blue'),
            (self.data['angles_gamma'], 'γ (gamma)', 'green')
        ]
        
        for ax, (angle, name, color) in zip(axes, angles_data):
            ax.hist(angle, bins=50, alpha=0.7, color=color, edgecolor='black', density=True)
            
            mean_val = np.mean(angle)
            std_val = np.std(angle)
            
            ax.axvline(mean_val, color='black', linestyle='--', linewidth=2,
                      label=f'Media: {mean_val:.3f}°')
            
            # Añadir distribución gaussiana ajustada
            x = np.linspace(angle.min(), angle.max(), 100)
            ax.plot(x, norm.pdf(x, mean_val, std_val), 'k-', linewidth=2,
                   alpha=0.6, label='Gaussiana')
            
            ax.set_xlabel(f'Ángulo {name} (grados)', fontsize=11)
            ax.set_ylabel('Densidad de probabilidad', fontsize=11)
            ax.set_title(f'{name}: {mean_val:.3f} ± {std_val:.3f}°', fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = self.output_dir / f"angles_histogram_T{int(self.config.temperature_K)}K.png"
        plt.savefig(filename, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        print(f"  💾 {filename.name}")
    
    def _plot_angle_evolution(self) -> None:
        """Genera gráfica de evolución temporal de ángulos"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(
            f'Evolución Temporal de Ángulos - AFI MD NPT\n'
            f'T={self.config.temperature_K}K, P={self.config.pressure_atm}atm',
            fontsize=14, fontweight='bold'
        )
        
        angles_data = [
            (self.data['angles_alpha'], 'α (alpha)', 'red'),
            (self.data['angles_beta'], 'β (beta)', 'blue'),
            (self.data['angles_gamma'], 'γ (gamma)', 'green')
        ]
        
        for ax, (angle, name, color) in zip(axes, angles_data):
            ax.plot(self.data['times'], angle, color=color, alpha=0.8, linewidth=1)
            
            mean_val = np.mean(angle)
            ax.axhline(mean_val, color='black', linestyle='--', linewidth=1.5,
                      label=f'Media: {mean_val:.3f}°')
            
            ax.set_ylabel(f'Ángulo {name} (°)', fontsize=11)
            ax.set_title(f'{name}: {mean_val:.3f} ± {np.std(angle):.3f}°', fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Tiempo (ps)', fontsize=11)
        plt.tight_layout()
        
        filename = self.output_dir / f"angles_evolution_T{int(self.config.temperature_K)}K.png"
        plt.savefig(filename, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        print(f"  💾 {filename.name}")
    
    def _plot_thermodynamics(self) -> None:
        """Genera gráfica de propiedades termodinámicas"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f'Propiedades Termodinámicas - AFI MD NPT\n'
            f'T={self.config.temperature_K}K, P={self.config.pressure_atm}atm',
            fontsize=14, fontweight='bold'
        )
        
        times = self.data['times']
        
        # Temperatura
        axes[0, 0].plot(times, self.data['temperatures'], color='orange', linewidth=1)
        axes[0, 0].axhline(self.config.temperature_K, color='red', linestyle='--',
                          linewidth=2, label=f'Target: {self.config.temperature_K}K')
        axes[0, 0].set_xlabel('Tiempo (ps)', fontsize=11)
        axes[0, 0].set_ylabel('Temperatura (K)', fontsize=11)
        mean_T = np.mean(self.data['temperatures'])
        std_T = np.std(self.data['temperatures'])
        axes[0, 0].set_title(f'Temperatura: {mean_T:.2f} ± {std_T:.2f} K')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Volumen
        axes[0, 1].plot(times, self.data['volumes'], color='purple', linewidth=1)
        axes[0, 1].set_xlabel('Tiempo (ps)', fontsize=11)
        axes[0, 1].set_ylabel('Volumen (Å³)', fontsize=11)
        mean_V = np.mean(self.data['volumes'])
        std_V = np.std(self.data['volumes'])
        axes[0, 1].set_title(f'Volumen: {mean_V:.2f} ± {std_V:.2f} Å³')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Energía potencial
        axes[1, 0].plot(times, self.data['energies_pot'], color='blue', linewidth=1)
        axes[1, 0].set_xlabel('Tiempo (ps)', fontsize=11)
        axes[1, 0].set_ylabel('Energía Potencial (eV)', fontsize=11)
        mean_Ep = np.mean(self.data['energies_pot'])
        std_Ep = np.std(self.data['energies_pot'])
        axes[1, 0].set_title(f'E_pot: {mean_Ep:.4f} ± {std_Ep:.4f} eV')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Energía total
        energies_total = self.data['energies_pot'] + self.data['energies_kin']
        axes[1, 1].plot(times, energies_total, color='green', linewidth=1)
        axes[1, 1].set_xlabel('Tiempo (ps)', fontsize=11)
        axes[1, 1].set_ylabel('Energía Total (eV)', fontsize=11)
        mean_Et = np.mean(energies_total)
        std_Et = np.std(energies_total)
        axes[1, 1].set_title(f'E_total: {mean_Et:.4f} ± {std_Et:.4f} eV')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = self.output_dir / f"thermodynamics_T{int(self.config.temperature_K)}K.png"
        plt.savefig(filename, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        print(f"  💾 {filename.name}")
    
    def print_summary(self) -> None:
        """Imprime resumen final de la simulación"""
        print_header("✅ SIMULACIÓN COMPLETADA")
        
        print(f"\n📁 Archivos generados en: {self.output_dir}/")
        print("\n📊 Datos:")
        print(f"  • afi_md_data_T{int(self.config.temperature_K)}K.npz (comprimido)")
        print(f"  • afi_md_data_T{int(self.config.temperature_K)}K.txt (texto)")
        
        print("\n🎬 Trayectorias:")
        print(f"  • equilibration_T{int(self.config.temperature_K)}K.traj")
        print(f"  • production_T{int(self.config.temperature_K)}K_P{self.config.pressure_atm}atm.traj")
        
        print("\n📝 Logs:")
        print(f"  • equilibration_T{int(self.config.temperature_K)}K.log")
        print(f"  • production_T{int(self.config.temperature_K)}K_P{self.config.pressure_atm}atm.log")
        
        if self.config.generate_plots:
            print("\n📈 Gráficas:")
            print(f"  • angles_histogram_T{int(self.config.temperature_K)}K.png")
            print(f"  • angles_evolution_T{int(self.config.temperature_K)}K.png")
            print(f"  • thermodynamics_T{int(self.config.temperature_K)}K.png")
        
        print("\n" + "="*70)
        
        # Sugerencias de análisis
        print("\n💡 Análisis adicional sugerido:")
        print("  • Visualizar trayectoria: ase gui production_*.traj")
        print("  • Cargar datos en Python:")
        print(f"    data = np.load('{self.output_dir}/afi_md_data_T{int(self.config.temperature_K)}K.npz')")
        print("    times = data['times']")
        print("    angles_alpha = data['angles_alpha']")
        print("\n" + "="*70 + "\n")


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal del script"""
    
    # Crear configuración
    config = SimulationConfig()
    
    # Crear simulación
    sim = AFIMDSimulation(config)
    
    # Setup
    if not sim.setup():
        print("\n❌ Error en la configuración inicial. Abortando.")
        sys.exit(1)
    
    # Fase 1: Equilibración
    if not sim.run_equilibration():
        print("\n❌ Error en la equilibración. Abortando.")
        sys.exit(1)
    
    # Fase 2: Producción
    if not sim.run_production():
        print("\n❌ Error en la producción. Abortando.")
        sys.exit(1)
    
    # Análisis y guardado
    sim.analyze_and_save()
    
    # Generar gráficas
    sim.generate_plots()
    
    # Resumen final
    sim.print_summary()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulación interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    def _plot_angle_evolution(self) -> None:
        """Genera gráfica de evolución temporal de ángulos"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(
            f'Evolución Temporal de Ángulos - AFI MD NPT\n'
            f'T={self.config.temperature_K}K, P={self.config.pressure_atm}atm',
            fontsize=14, fontweight='bold'
        )
        
        angles_data = [
            (self.data['angles_alpha'], 'α (alpha)', 'red'),
            (self.data['angles_beta'], 'β (beta)', 'blue'),
            (self.data['angles_gamma'], 'γ (gamma)', 'green')]
        
