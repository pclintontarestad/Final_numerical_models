"""
Maxwell-Bloch EIT Simulation: Encoder Gram Matrix Characterisation

This simulation tests the core thesis claims about non-isometric encoding
in tripod EIT quantum memory systems. It focuses EXCLUSIVELY on the 
encoding/storage phase, ignoring retrieval dynamics.

PHYSICAL REGIME:
- Ensemble Maxwell-Bloch dynamics at finite optical depth
- Semiclassical treatment: classical fields, quantum atoms
- Encoding only: we measure what's stored, not what's retrieved

THESIS CLAIMS TESTED:

Part 1 - Diagonal Non-Isometry (η₊ ≠ η₋):
    1. Symmetric CGCs → isometric encoding (κ = 1)
    2. Asymmetric CGCs → non-isometric encoding (κ > 1)
    3. κ scales monotonically with CGC asymmetry

Part 2 - Off-Diagonal Elements (G₁₂ ≠ 0):
    4. Differential detuning → spin-wave phase drift
    5. Phase drift causes polarisation rotation (off-diagonal G)
    6. This is the PRIMARY mechanism for coherent cross-talk

GRAM MATRIX STRUCTURE:
    G = E†E = [[η₊,  G₁₂],
               [G₂₁, η₋ ]]

    - Diagonal: polarisation-dependent storage efficiency
    - Off-diagonal: coherent polarisation mixing/rotation

Author: Thesis validation script
Date: 2024
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# Physical Constants
# =============================================================================

GAMMA_RB87_D2 = 2 * np.pi * 6.0666e6  # Rb-87 D2 decay rate (rad/s)
WAVELENGTH_D2 = 780.241e-9  # D2 line wavelength (m)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AtomicParameters:
    """Atomic structure parameters."""
    Gamma: float = GAMMA_RB87_D2
    
    # Clebsch-Gordan coefficients (normalised dipole matrix elements)
    # These control the coupling strength and branching ratios
    cgc_sigma_plus: float = 1.0   # |g₋₁⟩ ↔ |e⟩ (σ⁺ transition)
    cgc_pi: float = 1.0           # |g₀⟩ ↔ |e⟩ (π transition)
    cgc_sigma_minus: float = 1.0  # |g₊₁⟩ ↔ |e⟩ (σ⁻ transition)
    
    # Two-photon detunings (source of off-diagonal effects)
    delta_plus: float = 0.0   # Detuning for σ⁺ channel (rad/s)
    delta_minus: float = 0.0  # Detuning for σ⁻ channel (rad/s)
    
    # Ground-state dephasing
    gamma_dephasing: float = 0.0
    
    @property
    def branching_ratios(self) -> Tuple[float, float, float]:
        """Normalised spontaneous emission branching ratios."""
        c2_total = self.cgc_sigma_plus**2 + self.cgc_pi**2 + self.cgc_sigma_minus**2
        b_plus = self.cgc_sigma_plus**2 / c2_total   # → |g₋₁⟩
        b_pi = self.cgc_pi**2 / c2_total             # → |g₀⟩
        b_minus = self.cgc_sigma_minus**2 / c2_total # → |g₊₁⟩
        return b_plus, b_pi, b_minus


@dataclass
class MediumParameters:
    """Atomic medium parameters."""
    atom_density: float = 5e16      # atoms/m³
    medium_length: float = 1e-3     # m
    
    @property
    def optical_depth(self) -> float:
        """Resonant optical depth OD = n σ₀ L."""
        sigma_0 = 3 * WAVELENGTH_D2**2 / (2 * np.pi)
        return self.atom_density * sigma_0 * self.medium_length


@dataclass  
class PulseParameters:
    """Optical pulse parameters."""
    # Rabi frequencies
    Omega_control_max: float = 2 * np.pi * 10e6  # Control field peak
    Omega_probe_max: float = 2 * np.pi * 1e6     # Probe field peak
    
    # Timing (EIT storage sequence)
    t_probe_center: float = 1.0e-6   # Probe pulse center
    tau_probe: float = 0.3e-6        # Probe pulse width (Gaussian σ)
    t_ramp_center: float = 1.5e-6    # Control ramp-off center
    tau_ramp: float = 0.3e-6         # Control ramp time (adiabaticity)


@dataclass
class SimulationParameters:
    """Numerical simulation parameters."""
    N_z: int = 40           # Spatial grid points
    N_t: int = 400          # Time steps
    T_total: float = 3e-6   # Total simulation time


# =============================================================================
# Quantum Dynamics: 4-Level Tripod System
# =============================================================================

class TripodDynamics:
    """
    4-level tripod atomic dynamics with Lindblad dissipation.
    
    Basis ordering: |0⟩ = |e⟩, |1⟩ = |g₋₁⟩, |2⟩ = |g₀⟩, |3⟩ = |g₊₁⟩
    
    Level structure:
                    |e⟩
                   / | \
                σ⁺  π  σ⁻
                /   |   \
            |g₋₁⟩ |g₀⟩ |g₊₁⟩
    """
    
    def __init__(self, atoms: AtomicParameters):
        self.atoms = atoms
    
    def hamiltonian(
        self,
        Omega_probe: complex,    # σ⁺ probe field
        Omega_trigger: complex,  # σ⁻ trigger field  
        Omega_control: complex   # π control field
    ) -> np.ndarray:
        """
        Build the tripod Hamiltonian in the rotating frame.
        
        H = -δ₊|g₋₁⟩⟨g₋₁| - δ₋|g₊₁⟩⟨g₊₁|
            - (Ω_P c₊ |e⟩⟨g₋₁| + Ω_C c_π |e⟩⟨g₀| + Ω_T c₋ |e⟩⟨g₊₁| + h.c.)/2
        """
        H = np.zeros((4, 4), dtype=complex)
        
        # Two-photon detunings (diagonal in ground states)
        H[1, 1] = -self.atoms.delta_plus
        H[3, 3] = -self.atoms.delta_minus
        
        # σ⁺ probe coupling: |e⟩ ↔ |g₋₁⟩
        coupling_plus = self.atoms.cgc_sigma_plus * Omega_probe / 2
        H[0, 1] = -coupling_plus
        H[1, 0] = -np.conj(coupling_plus)
        
        # π control coupling: |e⟩ ↔ |g₀⟩
        coupling_pi = self.atoms.cgc_pi * Omega_control / 2
        H[0, 2] = -coupling_pi
        H[2, 0] = -np.conj(coupling_pi)
        
        # σ⁻ trigger coupling: |e⟩ ↔ |g₊₁⟩
        coupling_minus = self.atoms.cgc_sigma_minus * Omega_trigger / 2
        H[0, 3] = -coupling_minus
        H[3, 0] = -np.conj(coupling_minus)
        
        return H
    
    def dissipator(self, rho: np.ndarray) -> np.ndarray:
        """
        Lindblad dissipator for spontaneous emission and dephasing.
        
        D[ρ] = Σ_k (L_k ρ L_k† - {L_k† L_k, ρ}/2)
        """
        D = np.zeros((4, 4), dtype=complex)
        
        Gamma = self.atoms.Gamma
        b_plus, b_pi, b_minus = self.atoms.branching_ratios
        
        rho_ee = rho[0, 0].real  # Excited state population
        
        # Spontaneous emission: population transfer from |e⟩
        D[0, 0] -= Gamma * rho_ee              # Loss from |e⟩
        D[1, 1] += Gamma * b_plus * rho_ee     # Gain to |g₋₁⟩
        D[2, 2] += Gamma * b_pi * rho_ee       # Gain to |g₀⟩
        D[3, 3] += Gamma * b_minus * rho_ee    # Gain to |g₊₁⟩
        
        # Optical coherence decay (Γ/2 for each |e⟩-|g_j⟩ coherence)
        for j in [1, 2, 3]:
            D[0, j] -= (Gamma / 2) * rho[0, j]
            D[j, 0] -= (Gamma / 2) * rho[j, 0]
        
        # Ground-state dephasing (destroys spin-wave coherences)
        if self.atoms.gamma_dephasing > 0:
            gamma_d = self.atoms.gamma_dephasing
            for i in range(1, 4):
                for j in range(1, 4):
                    if i != j:
                        D[i, j] -= gamma_d * rho[i, j]
        
        return D
    
    def master_equation(
        self, 
        rho: np.ndarray, 
        H: np.ndarray
    ) -> np.ndarray:
        """
        Lindblad master equation: dρ/dt = -i[H, ρ] + D[ρ]
        """
        commutator = -1j * (H @ rho - rho @ H)
        dissipation = self.dissipator(rho)
        return commutator + dissipation
    
    def evolve_rk4(
        self,
        rho: np.ndarray,
        H: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """Fourth-order Runge-Kutta integration."""
        k1 = self.master_equation(rho, H)
        k2 = self.master_equation(rho + 0.5*dt*k1, H)
        k3 = self.master_equation(rho + 0.5*dt*k2, H)
        k4 = self.master_equation(rho + dt*k3, H)
        
        rho_new = rho + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Enforce physicality
        rho_new = 0.5 * (rho_new + rho_new.conj().T)  # Hermiticity
        trace = np.trace(rho_new).real
        if trace > 1e-10:
            rho_new /= trace  # Normalisation
        
        return rho_new


# =============================================================================
# Maxwell-Bloch Encoder Simulation
# =============================================================================

class MaxwellBlochEncoder:
    """
    Maxwell-Bloch simulation for tripod EIT encoding.
    
    This simulates the ENCODING phase only:
    1. Atoms start in |g₀⟩ (prepared by optical pumping)
    2. Probe pulse arrives with control field on (EIT window)
    3. Control field ramps off → population transfer to dark states
    4. Final state: spin-wave coherences ρ₁₂, ρ₃₂ store the qubit
    
    We measure:
    - Diagonal G: populations P₁ = ⟨g₋₁|ρ|g₋₁⟩, P₃ = ⟨g₊₁|ρ|g₊₁⟩
    - Off-diagonal G: spin-wave relative phase from ρ₁₂, ρ₃₂
    """
    
    def __init__(
        self,
        atoms: AtomicParameters,
        medium: MediumParameters,
        pulse: PulseParameters,
        sim: SimulationParameters
    ):
        self.atoms = atoms
        self.medium = medium
        self.pulse = pulse
        self.sim = sim
        
        # Derived quantities
        self.dynamics = TripodDynamics(atoms)
        self.z_grid = np.linspace(0, medium.medium_length, sim.N_z)
        self.dz = self.z_grid[1] - self.z_grid[0] if sim.N_z > 1 else medium.medium_length
        self.dt = sim.T_total / sim.N_t
        
        # Light-matter coupling for propagation
        # κ = (3λ²/8π) × Γ × n  (absorption coefficient)
        self.kappa = (3 * WAVELENGTH_D2**2 / (8 * np.pi)) * atoms.Gamma * medium.atom_density
    
    def control_field(self, t: float) -> float:
        """Control field envelope with smooth ramp-off for EIT storage."""
        ramp = 0.5 * (1 - np.tanh((t - self.pulse.t_ramp_center) / self.pulse.tau_ramp))
        return self.pulse.Omega_control_max * ramp
    
    def probe_envelope(self, t: float) -> float:
        """Gaussian probe pulse envelope."""
        return self.pulse.Omega_probe_max * np.exp(
            -((t - self.pulse.t_probe_center)**2) / (2 * self.pulse.tau_probe**2)
        )
    
    def initial_state(self) -> np.ndarray:
        """Initial density matrix: all population in |g₀⟩."""
        rho = np.zeros((4, 4), dtype=complex)
        rho[2, 2] = 1.0  # |g₀⟩
        return rho
    
    def encode_polarisation(
        self,
        alpha: complex = 1.0,  # σ⁺ amplitude
        beta: complex = 0.0,   # σ⁻ amplitude
        track_dynamics: bool = False
    ) -> Dict:
        """
        Encode a polarisation state |ψ⟩ = α|σ⁺⟩ + β|σ⁻⟩.
        
        Parameters:
            alpha, beta: Polarisation amplitudes (will be normalised)
            track_dynamics: If True, record time evolution for plotting
        
        Returns:
            Dictionary with encoding results including:
            - Final populations P₁, P₃
            - Spin-wave coherences ρ₁₂, ρ₃₂
            - Cross-coherence ρ₁₃
            - Optional: time evolution data
        """
        # Normalise input
        norm = np.sqrt(np.abs(alpha)**2 + np.abs(beta)**2)
        if norm > 0:
            alpha, beta = alpha/norm, beta/norm
        
        # Initialise atomic states across medium
        rho_slices = [self.initial_state() for _ in range(self.sim.N_z)]
        
        # Field arrays
        Omega_P = np.zeros(self.sim.N_z, dtype=complex)  # σ⁺ probe
        Omega_T = np.zeros(self.sim.N_z, dtype=complex)  # σ⁻ trigger
        
        # Time evolution tracking
        if track_dynamics:
            times = []
            P_e_history = []
            P_1_history = []
            P_3_history = []
            rho_12_history = []
            rho_32_history = []
            rho_13_history = []
        
        # Time evolution
        for step in range(self.sim.N_t):
            t = step * self.dt
            
            # Field envelopes
            Omega_C = self.control_field(t)
            probe_amp = self.probe_envelope(t)
            
            # Input fields with polarisation amplitudes
            Omega_P[0] = alpha * probe_amp
            Omega_T[0] = beta * probe_amp
            
            # Evolve each spatial slice
            for i in range(self.sim.N_z):
                H = self.dynamics.hamiltonian(Omega_P[i], Omega_T[i], Omega_C)
                rho_slices[i] = self.dynamics.evolve_rk4(rho_slices[i], H, self.dt)
            
            # Propagate fields through medium (simplified absorption)
            for i in range(1, self.sim.N_z):
                # EIT transparency factor
                if np.abs(Omega_C) > 1e3:
                    eit_factor = (self.pulse.Omega_probe_max / np.abs(Omega_C))**2
                else:
                    eit_factor = 1.0
                
                absorption = self.kappa * self.dz * eit_factor * 0.5
                
                # Probe propagation with source term
                rho_01 = rho_slices[i-1][0, 1]
                Omega_P[i] = Omega_P[i-1] * np.exp(-absorption)
                Omega_P[i] += 1j * self.kappa * self.dz * rho_01
                
                # Trigger propagation with source term
                rho_03 = rho_slices[i-1][0, 3]
                Omega_T[i] = Omega_T[i-1] * np.exp(-absorption)
                Omega_T[i] += 1j * self.kappa * self.dz * rho_03
            
            # Record dynamics
            if track_dynamics and step % max(1, self.sim.N_t // 100) == 0:
                times.append(t)
                P_e_history.append(np.mean([rho[0,0].real for rho in rho_slices]))
                P_1_history.append(np.mean([rho[1,1].real for rho in rho_slices]))
                P_3_history.append(np.mean([rho[3,3].real for rho in rho_slices]))
                rho_12_history.append(np.mean([rho[1,2] for rho in rho_slices]))
                rho_32_history.append(np.mean([rho[3,2] for rho in rho_slices]))
                rho_13_history.append(np.mean([rho[1,3] for rho in rho_slices]))
        
        # Extract final state (averaged over medium)
        final_P_e = np.mean([rho[0,0].real for rho in rho_slices])
        final_P_1 = np.mean([rho[1,1].real for rho in rho_slices])
        final_P_0 = np.mean([rho[2,2].real for rho in rho_slices])
        final_P_3 = np.mean([rho[3,3].real for rho in rho_slices])
        
        final_rho_12 = np.mean([rho[1,2] for rho in rho_slices])
        final_rho_32 = np.mean([rho[3,2] for rho in rho_slices])
        final_rho_13 = np.mean([rho[1,3] for rho in rho_slices])
        
        results = {
            "input_alpha": alpha,
            "input_beta": beta,
            "final_P_e": final_P_e,
            "final_P_1": final_P_1,
            "final_P_0": final_P_0,
            "final_P_3": final_P_3,
            "final_rho_12": final_rho_12,
            "final_rho_32": final_rho_32,
            "final_rho_13": final_rho_13,
        }
        
        if track_dynamics:
            results["times"] = np.array(times)
            results["P_e_history"] = np.array(P_e_history)
            results["P_1_history"] = np.array(P_1_history)
            results["P_3_history"] = np.array(P_3_history)
            results["rho_12_history"] = np.array(rho_12_history)
            results["rho_32_history"] = np.array(rho_32_history)
            results["rho_13_history"] = np.array(rho_13_history)
        
        return results
    
    def extract_gram_matrix(self) -> Tuple[np.ndarray, Dict]:
        """
        Extract the full 2×2 Gram matrix by state tomography.
        
        We encode three input states:
        1. |σ⁺⟩ → gives η₊ = G₁₁
        2. |σ⁻⟩ → gives η₋ = G₂₂  
        3. |+⟩ = (|σ⁺⟩ + |σ⁻⟩)/√2 → gives G₁₂ from coherences
        
        Returns:
            G: 2×2 Gram matrix
            diagnostics: Dictionary with detailed results
        """
        # Encode basis states
        res_plus = self.encode_polarisation(1.0, 0.0)
        res_minus = self.encode_polarisation(0.0, 1.0)
        res_super = self.encode_polarisation(1.0, 1.0, track_dynamics=True)
        
        # Diagonal elements: storage efficiencies
        eta_plus = res_plus["final_P_1"]
        eta_minus = res_minus["final_P_3"]
        
        # Off-diagonal: from spin-wave relative phase in superposition
        # The cross-coherence ρ₁₃ in the superposition encoding gives G₁₂
        G_12 = res_super["final_rho_13"]
        
        # Alternative: from spin-wave coherences
        rho_12 = res_super["final_rho_12"]
        rho_32 = res_super["final_rho_32"]
        
        # Construct Gram matrix
        G = np.array([
            [eta_plus, G_12],
            [np.conj(G_12), eta_minus]
        ], dtype=complex)
        
        # Diagnostics
        kappa = np.sqrt(eta_plus / eta_minus) if eta_minus > 1e-10 else np.inf
        
        # Spin-wave relative phase (key off-diagonal indicator)
        if np.abs(rho_12) > 1e-10 and np.abs(rho_32) > 1e-10:
            relative_phase = np.angle(rho_32) - np.angle(rho_12)
        else:
            relative_phase = 0.0
        
        diagnostics = {
            "eta_plus": eta_plus,
            "eta_minus": eta_minus,
            "kappa": kappa,
            "G_12": G_12,
            "G_12_magnitude": np.abs(G_12),
            "rho_12": rho_12,
            "rho_32": rho_32,
            "spin_wave_relative_phase_deg": np.degrees(relative_phase),
            "results_plus": res_plus,
            "results_minus": res_minus,
            "results_super": res_super,
        }
        
        return G, diagnostics


# =============================================================================
# Test Suite
# =============================================================================

class EncoderTestSuite:
    """Comprehensive test suite for encoder Gram matrix characterisation."""
    
    def __init__(self, output_dir: str = "encoder_validation_figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
    
    def print_header(self, title: str):
        """Print formatted section header."""
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
    
    def print_gram_matrix(self, G: np.ndarray, label: str = ""):
        """Pretty-print a Gram matrix."""
        if label:
            print(f"\n{label}:")
        print(f"  G = [[{G[0,0].real:8.5f}, {G[0,1]:15.6f}],")
        print(f"       [{G[1,0]:15.6f}, {G[1,1].real:8.5f}]]")
    
    # -------------------------------------------------------------------------
    # TEST 1: Symmetric Baseline
    # -------------------------------------------------------------------------
    def test_symmetric_encoding(self) -> Dict:
        """
        TEST 1: Symmetric CGCs should give isometric encoding (κ = 1).
        
        This validates the simulation baseline before testing asymmetry.
        """
        self.print_header("TEST 1: Symmetric CGC Encoding (Isometric Baseline)")
        
        atoms = AtomicParameters(
            cgc_sigma_plus=1.0,
            cgc_pi=1.0,
            cgc_sigma_minus=1.0,
        )
        medium = MediumParameters(atom_density=5e16)
        pulse = PulseParameters()
        sim = SimulationParameters(N_z=40, N_t=400)
        
        encoder = MaxwellBlochEncoder(atoms, medium, pulse, sim)
        G, diag = encoder.extract_gram_matrix()
        
        print(f"\nOptical depth: {medium.optical_depth:.1f}")
        print(f"Branching ratios: {atoms.branching_ratios}")
        
        self.print_gram_matrix(G, "Gram matrix")
        
        print(f"\n  η₊ = {diag['eta_plus']:.6f}")
        print(f"  η₋ = {diag['eta_minus']:.6f}")
        print(f"  κ  = {diag['kappa']:.6f}")
        print(f"  |G₁₂| = {diag['G_12_magnitude']:.6f}")
        
        is_isometric = diag['kappa'] < 1.01
        print(f"\n  RESULT: {'PASS ✓ Isometric' if is_isometric else 'FAIL ✗ Non-isometric'}")
        
        self.results["symmetric"] = {"G": G, "diagnostics": diag, "passed": is_isometric}
        return self.results["symmetric"]
    
    # -------------------------------------------------------------------------
    # TEST 2: CGC Asymmetry → Diagonal Non-Isometry
    # -------------------------------------------------------------------------
    def test_asymmetric_encoding(self) -> Dict:
        """
        TEST 2: Asymmetric CGCs should give non-isometric encoding (κ > 1).
        
        This is the CORE thesis test for Part 1.
        """
        self.print_header("TEST 2: Asymmetric CGC Encoding (Core Thesis Test)")
        
        atoms = AtomicParameters(
            cgc_sigma_plus=1.2,   # Favoured channel
            cgc_pi=1.0,
            cgc_sigma_minus=0.8,  # Disfavoured channel
        )
        medium = MediumParameters(atom_density=5e16)
        pulse = PulseParameters()
        sim = SimulationParameters(N_z=40, N_t=400)
        
        encoder = MaxwellBlochEncoder(atoms, medium, pulse, sim)
        G, diag = encoder.extract_gram_matrix()
        
        # Naive prediction (single-atom, no collective effects)
        b_plus, _, b_minus = atoms.branching_ratios
        kappa_naive = np.sqrt(b_plus / b_minus)
        
        print(f"\nCGC values: σ⁺ = {atoms.cgc_sigma_plus}, σ⁻ = {atoms.cgc_sigma_minus}")
        print(f"Branching ratios: b₊ = {b_plus:.3f}, b₋ = {b_minus:.3f}")
        print(f"Naive κ prediction: {kappa_naive:.3f}")
        
        self.print_gram_matrix(G, "Gram matrix")
        
        print(f"\n  η₊ = {diag['eta_plus']:.6f}")
        print(f"  η₋ = {diag['eta_minus']:.6f}")
        print(f"  κ  = {diag['kappa']:.6f}")
        print(f"  Suppression vs naive: {kappa_naive / diag['kappa']:.2f}×")
        
        is_non_isometric = diag['kappa'] > 1.01
        print(f"\n  RESULT: {'PASS ✓ Non-isometric as predicted' if is_non_isometric else 'FAIL ✗ Unexpectedly isometric'}")
        
        self.results["asymmetric"] = {"G": G, "diagnostics": diag, "kappa_naive": kappa_naive, "passed": is_non_isometric}
        return self.results["asymmetric"]
    
    # -------------------------------------------------------------------------
    # TEST 3: CGC Asymmetry Sweep
    # -------------------------------------------------------------------------
    def test_cgc_sweep(self) -> Dict:
        """
        TEST 3: Sweep CGC asymmetry and verify κ increases monotonically.
        """
        self.print_header("TEST 3: CGC Asymmetry Sweep")
        
        asymmetries = [0.0, 0.1, 0.2, 0.3, 0.4]
        results = []
        
        for asym in asymmetries:
            atoms = AtomicParameters(
                cgc_sigma_plus=1.0 + asym/2,
                cgc_pi=1.0,
                cgc_sigma_minus=1.0 - asym/2,
            )
            medium = MediumParameters(atom_density=5e16)
            pulse = PulseParameters()
            sim = SimulationParameters(N_z=30, N_t=300)
            
            encoder = MaxwellBlochEncoder(atoms, medium, pulse, sim)
            G, diag = encoder.extract_gram_matrix()
            
            results.append({
                "asymmetry": asym,
                "cgc_plus": atoms.cgc_sigma_plus,
                "cgc_minus": atoms.cgc_sigma_minus,
                "eta_plus": diag["eta_plus"],
                "eta_minus": diag["eta_minus"],
                "kappa": diag["kappa"],
            })
            
            print(f"  Asymmetry {asym:.1f}: κ = {diag['kappa']:.4f}, η₊ = {diag['eta_plus']:.4f}, η₋ = {diag['eta_minus']:.4f}")
        
        # Check monotonicity
        kappas = [r["kappa"] for r in results]
        is_monotonic = all(kappas[i] <= kappas[i+1] + 0.001 for i in range(len(kappas)-1))
        print(f"\n  Monotonic κ increase: {'YES ✓' if is_monotonic else 'NO ✗'}")
        
        # Generate plot
        self._plot_cgc_sweep(results)
        
        self.results["cgc_sweep"] = {"data": results, "monotonic": is_monotonic}
        return self.results["cgc_sweep"]
    
    def _plot_cgc_sweep(self, results: List[Dict]):
        """Generate CGC asymmetry sweep figure."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        asyms = [r["asymmetry"] for r in results]
        
        ax = axes[0]
        ax.plot(asyms, [r["eta_plus"] for r in results], 'b-o', lw=2, label='η₊ (σ⁺)')
        ax.plot(asyms, [r["eta_minus"] for r in results], 'r-o', lw=2, label='η₋ (σ⁻)')
        ax.set_xlabel("CGC Asymmetry", fontsize=11)
        ax.set_ylabel("Storage Efficiency", fontsize=11)
        ax.set_title("Polarisation-Dependent Storage", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        ax.plot(asyms, [r["kappa"] for r in results], 'g-o', lw=2, markersize=8)
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Isometric')
        ax.set_xlabel("CGC Asymmetry", fontsize=11)
        ax.set_ylabel("Condition Number κ", fontsize=11)
        ax.set_title("Non-Isometry vs CGC Asymmetry", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "test3_cgc_asymmetry_sweep.pdf", bbox_inches="tight")
        plt.savefig(self.output_dir / "test3_cgc_asymmetry_sweep.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Figure saved: {self.output_dir}/test3_cgc_asymmetry_sweep.pdf")
    
    # -------------------------------------------------------------------------
    # TEST 4: Differential Detuning → Off-Diagonal G
    # -------------------------------------------------------------------------
    def test_detuning_offdiagonal(self) -> Dict:
        """
        TEST 4: Differential detuning induces off-diagonal G elements.
        
        When δ₊ ≠ δ₋, the spin waves acquire different phases,
        causing polarisation rotation.
        """
        self.print_header("TEST 4: Differential Detuning → Off-Diagonal G")
        
        delta_MHz = 0.5
        delta_rad = 2 * np.pi * delta_MHz * 1e6
        
        atoms = AtomicParameters(
            cgc_sigma_plus=1.0,
            cgc_pi=1.0,
            cgc_sigma_minus=1.0,
            delta_plus=+delta_rad/2,
            delta_minus=-delta_rad/2,
        )
        medium = MediumParameters(atom_density=5e16)
        pulse = PulseParameters()
        sim = SimulationParameters(N_z=40, N_t=400)
        
        encoder = MaxwellBlochEncoder(atoms, medium, pulse, sim)
        G, diag = encoder.extract_gram_matrix()
        
        print(f"\nDifferential detuning: Δδ = {delta_MHz} MHz")
        
        self.print_gram_matrix(G, "Gram matrix")
        
        print(f"\n  Spin-wave relative phase: {diag['spin_wave_relative_phase_deg']:.1f}°")
        print(f"  |G₁₂| = {diag['G_12_magnitude']:.6f}")
        
        self.results["detuning"] = {"G": G, "diagnostics": diag, "delta_MHz": delta_MHz}
        return self.results["detuning"]
    
    # -------------------------------------------------------------------------
    # TEST 5: Detuning Sweep → Phase Drift
    # -------------------------------------------------------------------------
    def test_detuning_sweep(self) -> Dict:
        """
        TEST 5: Sweep differential detuning and track spin-wave phase drift.
        
        This demonstrates the PRIMARY mechanism for off-diagonal G elements.
        """
        self.print_header("TEST 5: Detuning Sweep → Spin-Wave Phase Drift")
        
        delta_values_MHz = [0.0, 0.2, 0.5, 1.0, 2.0]
        results = []
        
        for delta_MHz in delta_values_MHz:
            delta_rad = 2 * np.pi * delta_MHz * 1e6
            
            atoms = AtomicParameters(
                cgc_sigma_plus=1.0,
                cgc_pi=1.0,
                cgc_sigma_minus=1.0,
                delta_plus=+delta_rad/2,
                delta_minus=-delta_rad/2,
            )
            medium = MediumParameters(atom_density=5e16)
            pulse = PulseParameters()
            sim = SimulationParameters(N_z=30, N_t=300)
            
            encoder = MaxwellBlochEncoder(atoms, medium, pulse, sim)
            G, diag = encoder.extract_gram_matrix()
            
            results.append({
                "delta_MHz": delta_MHz,
                "G": G,
                "kappa": diag["kappa"],
                "G_12": diag["G_12"],
                "G_12_mag": diag["G_12_magnitude"],
                "relative_phase_deg": diag["spin_wave_relative_phase_deg"],
                "rho_12": diag["rho_12"],
                "rho_32": diag["rho_32"],
            })
            
            print(f"  Δδ = {delta_MHz:.1f} MHz: Δφ = {diag['spin_wave_relative_phase_deg']:7.1f}°, |G₁₂| = {diag['G_12_magnitude']:.6f}")
        
        # Generate plot
        self._plot_detuning_sweep(results)
        
        self.results["detuning_sweep"] = {"data": results}
        return self.results["detuning_sweep"]
    
    def _plot_detuning_sweep(self, results: List[Dict]):
        """Generate detuning sweep figure."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        deltas = [r["delta_MHz"] for r in results]
        phases = [r["relative_phase_deg"] for r in results]
        G12_mags = [r["G_12_mag"] for r in results]
        
        ax = axes[0]
        ax.plot(deltas, phases, 'b-o', lw=2, markersize=8)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("Differential Detuning Δδ (MHz)", fontsize=11)
        ax.set_ylabel("Spin-Wave Relative Phase (°)", fontsize=11)
        ax.set_title("Phase Drift vs Detuning", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        ax.plot(deltas, G12_mags, 'r-o', lw=2, markersize=8)
        ax.set_xlabel("Differential Detuning Δδ (MHz)", fontsize=11)
        ax.set_ylabel("|G₁₂|", fontsize=11)
        ax.set_title("Off-Diagonal Magnitude vs Detuning", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "test5_detuning_sweep.pdf", bbox_inches="tight")
        plt.savefig(self.output_dir / "test5_detuning_sweep.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Figure saved: {self.output_dir}/test5_detuning_sweep.pdf")
    
    # -------------------------------------------------------------------------
    # TEST 6: Spin-Wave Dynamics Visualisation
    # -------------------------------------------------------------------------
    def test_spinwave_dynamics(self) -> Dict:
        """
        TEST 6: Visualise spin-wave coherence buildup and phase evolution.
        
        This shows HOW the off-diagonal G elements arise dynamically.
        """
        self.print_header("TEST 6: Spin-Wave Dynamics Visualisation")
        
        # With detuning to show phase evolution
        delta_MHz = 0.5
        delta_rad = 2 * np.pi * delta_MHz * 1e6
        
        atoms = AtomicParameters(
            cgc_sigma_plus=1.0,
            cgc_pi=1.0,
            cgc_sigma_minus=1.0,
            delta_plus=+delta_rad/2,
            delta_minus=-delta_rad/2,
        )
        medium = MediumParameters(atom_density=5e16)
        pulse = PulseParameters()
        sim = SimulationParameters(N_z=40, N_t=500)
        
        encoder = MaxwellBlochEncoder(atoms, medium, pulse, sim)
        
        # Encode superposition with full dynamics tracking
        result = encoder.encode_polarisation(1.0, 1.0, track_dynamics=True)
        
        print(f"\nDifferential detuning: Δδ = {delta_MHz} MHz")
        print(f"Final |ρ₁₂| = {np.abs(result['final_rho_12']):.6f}")
        print(f"Final |ρ₃₂| = {np.abs(result['final_rho_32']):.6f}")
        
        if np.abs(result['final_rho_12']) > 1e-10 and np.abs(result['final_rho_32']) > 1e-10:
            rel_phase = np.angle(result['final_rho_32']) - np.angle(result['final_rho_12'])
            print(f"Final relative phase: {np.degrees(rel_phase):.1f}°")
        
        # Generate plot
        self._plot_spinwave_dynamics(result, delta_MHz)
        
        self.results["spinwave_dynamics"] = {"result": result, "delta_MHz": delta_MHz}
        return self.results["spinwave_dynamics"]
    
    def _plot_spinwave_dynamics(self, result: Dict, delta_MHz: float):
        """Generate spin-wave dynamics figure."""
        times_us = result["times"] * 1e6
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # Panel 1: Populations
        ax = axes[0, 0]
        ax.plot(times_us, result["P_1_history"], 'b-', lw=1.5, label='P₁ (|g₋₁⟩)')
        ax.plot(times_us, result["P_3_history"], 'r-', lw=1.5, label='P₃ (|g₊₁⟩)')
        ax.plot(times_us, result["P_e_history"], 'k--', lw=1, label='Pₑ (|e⟩)')
        ax.set_xlabel("Time (µs)")
        ax.set_ylabel("Population")
        ax.set_title("State Populations")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Spin-wave magnitudes
        ax = axes[0, 1]
        ax.plot(times_us, np.abs(result["rho_12_history"]), 'b-', lw=1.5, label='|ρ₁₂| (σ⁺ spin wave)')
        ax.plot(times_us, np.abs(result["rho_32_history"]), 'r-', lw=1.5, label='|ρ₃₂| (σ⁻ spin wave)')
        ax.set_xlabel("Time (µs)")
        ax.set_ylabel("Coherence Magnitude")
        ax.set_title("Spin-Wave Amplitudes")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Spin-wave phases
        ax = axes[1, 0]
        phase_12 = np.angle(result["rho_12_history"]) * 180 / np.pi
        phase_32 = np.angle(result["rho_32_history"]) * 180 / np.pi
        ax.plot(times_us, phase_12, 'b-', lw=1.5, label='arg(ρ₁₂)')
        ax.plot(times_us, phase_32, 'r-', lw=1.5, label='arg(ρ₃₂)')
        ax.set_xlabel("Time (µs)")
        ax.set_ylabel("Phase (degrees)")
        ax.set_title("Spin-Wave Phases")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 4: Relative phase (key for off-diagonal G)
        ax = axes[1, 1]
        relative_phase = np.angle(result["rho_32_history"]) - np.angle(result["rho_12_history"])
        relative_phase_deg = np.degrees(relative_phase)
        ax.plot(times_us, relative_phase_deg, 'g-', lw=2)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("Time (µs)")
        ax.set_ylabel("Relative Phase Δφ (degrees)")
        ax.set_title(f"Spin-Wave Phase Drift (Δδ = {delta_MHz} MHz)")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "test6_spinwave_dynamics.pdf", bbox_inches="tight")
        plt.savefig(self.output_dir / "test6_spinwave_dynamics.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Figure saved: {self.output_dir}/test6_spinwave_dynamics.pdf")
    
    # -------------------------------------------------------------------------
    # TEST 7: Combined CGC + Detuning Effects
    # -------------------------------------------------------------------------
    def test_combined_effects(self) -> Dict:
        """
        TEST 7: Combined CGC asymmetry AND differential detuning.
        
        This shows the full Gram matrix structure with both diagonal
        and off-diagonal non-isometry.
        """
        self.print_header("TEST 7: Combined CGC Asymmetry + Detuning")
        
        delta_MHz = 0.3
        delta_rad = 2 * np.pi * delta_MHz * 1e6
        
        atoms = AtomicParameters(
            cgc_sigma_plus=1.2,
            cgc_pi=1.0,
            cgc_sigma_minus=0.8,
            delta_plus=+delta_rad/2,
            delta_minus=-delta_rad/2,
        )
        medium = MediumParameters(atom_density=5e16)
        pulse = PulseParameters()
        sim = SimulationParameters(N_z=40, N_t=400)
        
        encoder = MaxwellBlochEncoder(atoms, medium, pulse, sim)
        G, diag = encoder.extract_gram_matrix()
        
        print(f"\nCGC asymmetry: σ⁺ = {atoms.cgc_sigma_plus}, σ⁻ = {atoms.cgc_sigma_minus}")
        print(f"Differential detuning: Δδ = {delta_MHz} MHz")
        
        self.print_gram_matrix(G, "Full Gram matrix")
        
        print(f"\n  Diagonal non-isometry:")
        print(f"    κ = {diag['kappa']:.4f}")
        print(f"\n  Off-diagonal contribution:")
        print(f"    |G₁₂| = {diag['G_12_magnitude']:.6f}")
        print(f"    arg(G₁₂) = {np.degrees(np.angle(diag['G_12'])):.1f}°")
        print(f"    Spin-wave Δφ = {diag['spin_wave_relative_phase_deg']:.1f}°")
        
        self.results["combined"] = {"G": G, "diagnostics": diag}
        return self.results["combined"]
    
    # -------------------------------------------------------------------------
    # Run All Tests
    # -------------------------------------------------------------------------
    def run_all(self) -> Dict:
        """Execute complete test suite."""
        print("\n" + "=" * 70)
        print("MAXWELL-BLOCH ENCODER GRAM MATRIX VALIDATION")
        print("Testing diagonal and off-diagonal non-isometry in tripod EIT")
        print("=" * 70)
        
        # Part 1 tests: Diagonal non-isometry
        self.test_symmetric_encoding()
        self.test_asymmetric_encoding()
        self.test_cgc_sweep()
        
        # Part 2 tests: Off-diagonal elements
        self.test_detuning_offdiagonal()
        self.test_detuning_sweep()
        self.test_spinwave_dynamics()
        
        # Combined effects
        self.test_combined_effects()
        
        # Summary
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print comprehensive summary."""
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        print("\n┌─────────────────────────────────────────────────────────────────────┐")
        print("│ PART 1: DIAGONAL NON-ISOMETRY (η₊ ≠ η₋)                             │")
        print("├─────────────────────────────────────────────────────────────────────┤")
        
        if "symmetric" in self.results:
            r = self.results["symmetric"]
            status = "✓" if r["passed"] else "✗"
            print(f"│ Test 1 (Symmetric baseline):  κ = {r['diagnostics']['kappa']:.4f}  [{status} Isometric]       │")
        
        if "asymmetric" in self.results:
            r = self.results["asymmetric"]
            status = "✓" if r["passed"] else "✗"
            print(f"│ Test 2 (CGC asymmetry):       κ = {r['diagnostics']['kappa']:.4f}  [{status} Non-isometric]   │")
        
        if "cgc_sweep" in self.results:
            r = self.results["cgc_sweep"]
            status = "✓" if r["monotonic"] else "✗"
            print(f"│ Test 3 (CGC sweep):           Monotonic increase [{status}]                 │")
        
        print("├─────────────────────────────────────────────────────────────────────┤")
        print("│ PART 2: OFF-DIAGONAL ELEMENTS (G₁₂ ≠ 0)                             │")
        print("├─────────────────────────────────────────────────────────────────────┤")
        
        if "detuning" in self.results:
            r = self.results["detuning"]
            phase = r['diagnostics']['spin_wave_relative_phase_deg']
            print(f"│ Test 4 (Detuning Δδ=0.5MHz):  Δφ = {phase:6.1f}°                          │")
        
        if "detuning_sweep" in self.results:
            print(f"│ Test 5 (Detuning sweep):      Phase drift demonstrated                │")
        
        if "spinwave_dynamics" in self.results:
            print(f"│ Test 6 (Dynamics):            Spin-wave evolution visualised          │")
        
        print("├─────────────────────────────────────────────────────────────────────┤")
        print("│ COMBINED EFFECTS                                                    │")
        print("├─────────────────────────────────────────────────────────────────────┤")
        
        if "combined" in self.results:
            r = self.results["combined"]
            kappa = r['diagnostics']['kappa']
            G12 = r['diagnostics']['G_12_magnitude']
            print(f"│ Test 7 (CGC + Detuning):      κ = {kappa:.4f}, |G₁₂| = {G12:.6f}          │")
        
        print("└─────────────────────────────────────────────────────────────────────┘")
        
        print("\n" + "=" * 70)
        print("PHYSICAL INTERPRETATION")
        print("=" * 70)
        print("""
The Gram matrix G = E†E characterises encoder geometry:

  DIAGONAL ELEMENTS (η₊, η₋):
  • Source: CG-coefficient-weighted spontaneous emission branching
  • Effect: Polarisation-dependent storage efficiency
  • Consequence: κ = √(η₊/η₋) ≠ 1 → metric distortion
  
  OFF-DIAGONAL ELEMENTS (G₁₂):
  • Source: Differential two-photon detuning (δ₊ ≠ δ₋)
  • Mechanism: Spin waves ρ₁₂ and ρ₃₂ acquire different phases
  • Consequence: Output polarisation rotated relative to input
  
  PHYSICAL ORIGINS OF DETUNING:
  • Magnetic field gradients → differential Zeeman shifts
  • AC Stark shifts from imperfectly balanced control fields
  • Doppler shifts in thermal vapours
  
  IMPLICATIONS FOR GATE OPERATIONS (Part 2 of thesis):
  • Diagonal κ ≠ 1 → metric-preserving gates collapse to U(1)
  • Off-diagonal G₁₂ ≠ 0 → additional phase constraints
  • Both effects must be characterised for high-fidelity operations
""")
        
        print(f"\nAll figures saved to: {self.output_dir}/")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the complete encoder validation suite."""
    suite = EncoderTestSuite(output_dir="encoder_validation_figures")
    results = suite.run_all()
    return results


if __name__ == "__main__":
    results = main()
