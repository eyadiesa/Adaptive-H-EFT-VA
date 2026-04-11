#### **File: `src/adaptive_main.py` (Adaptive H-EFT-VA — Paper 2)**
####
#### Companion code to:
####   "Adaptive H-EFT-VA: Dynamic UV-Cutoff Relaxation for Global
####    Expressivity in Variational Quantum Algorithms"
####   Eyad I. B. Hamid (2026)
####
#### Builds directly on H-EFT-VA (arXiv:2601.10479).
#### All original infrastructure (ansatz, Hamiltonians, helpers, plotting
#### style) is preserved verbatim from main.py (Paper 1).
#### New additions are clearly marked with "# [A-H-EFT NEW]".
####
#### Tests in this file:
####   AT1  — Adaptive GV scaling: BP avoidance throughout both phases
####   AT2  — Critical cutoff sweep: sigma vs gradient variance at N=14
####   AT3  — Phase transition: GV across Phase I / switch / Phase II
####   AT4  — Adaptive convergence vs static H-EFT-VA vs HEA
####   AT5  — Convergence vs system size (adaptive)
####   AT6  — Ground-state fidelity: adaptive vs static vs HEA
####   AT7  — Reference-state gap Delta_ref vs N (TFIM and Heisenberg)
####   AT8  — Effective Hilbert space growth: deff vs t
####   AT9  — Entanglement entropy: Phase I vs Phase II
####   AT10 — Expressibility proxy: adaptive vs static vs HEA vs Haar
####   AT11 — Noise robustness of adaptive training
####   AT12 — Finite-shot gradient estimator under adaptive schedule
####   AT13 — Switch criterion sensitivity (delta_switch sweep)
####   AT14 — Growth constant sensitivity (lambda sweep)
####   AT15 — Statistical significance: adaptive vs all baselines
####   AT16 — Heisenberg XXZ: full adaptive benchmark

import os
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
import json
import seaborn as sns
import time
from typing import Callable, Tuple, List, Dict, Any
import pandas as pd
from scipy import stats

# ===========================================================================
# --- Configuration (mirrors Paper 1 exactly) ---
# ===========================================================================

QUBIT_LIST      = [2, 4, 6, 8, 10, 12, 14]
LAYER_LIST      = [2, 4, 6, 8, 10, 12, 14]
SEEDS           = range(50)
N_OPTIMIZER_STEPS = 100
OPTIMIZER_LR    = 0.01
HAMILTONIAN_NAME = 'tfim'
KAPPA           = 0.1          # Paper 1 EFT coupling-scale bound

# [A-H-EFT NEW] Adaptive-specific globals
LAMBDA_DEFAULT       = 0.02    # Growth constant for sigma(t) = sigma0 * exp(lambda*t)
DELTA_SWITCH_DEFAULT = 1e-3    # Gradient-norm threshold for Phase I -> Phase II
SIGMA_CRIT_C2        = 0.5     # Universal constant c2 in sigma_crit(N) = c2/sqrt(L*N)
N_ADAPTIVE_STEPS     = 200     # Total steps for adaptive runs (Phase I + Phase II)

RES_DIR = 'results_adaptive'
FIG_DIR = 'figures_adaptive'
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams["figure.dpi"] = 600
sns.set(style="whitegrid", context="paper", font_scale=1.2)

# ===========================================================================
# --- Ansatz Definitions (verbatim from Paper 1) ---
# ===========================================================================

def heft_va_layer(params: np.ndarray, wires: List[int],
                  p_noise: float = 0.0, mode: str = 'spin'):
    """
    H-EFT-VA Layer (Paper 1, preserved verbatim).
    'spin' mode: CNOT-Rz-CNOT for TFIM/Heisenberg.
    'chemistry' mode: IsingXY for orbital hopping.
    """
    n = len(wires)
    for i in range(n):
        qml.RY(params[i], wires=wires[i])
        if mode == 'chemistry':
            qml.RZ(params[n + i], wires=wires[i])
            if p_noise > 0:
                qml.DepolarizingChannel(p_noise / 10, wires=wires[i])

    offset = n if mode == 'spin' else 2 * n

    for i in range(n - 1):
        w1, w2 = wires[i], wires[i + 1]
        if mode == 'spin':
            qml.CNOT(wires=[w1, w2])
            qml.RZ(params[offset + i], wires=w2)
            qml.CNOT(wires=[w1, w2])
        else:
            qml.IsingXY(params[offset + i], wires=[w1, w2])
        if p_noise > 0:
            qml.DepolarizingChannel(p_noise, wires=w1)
            qml.DepolarizingChannel(p_noise, wires=w2)


def heft_va_ansatz(params: np.ndarray, n_qubits: int, n_layers: int,
                   p_noise: float = 0.0, mode: str = 'spin'):
    """Full H-EFT-VA circuit (Paper 1, preserved verbatim)."""
    params_per_layer = (n_qubits + (n_qubits - 1)) if mode == 'spin' \
                       else (2 * n_qubits + (n_qubits - 1))
    try:
        params = params.reshape((n_layers, params_per_layer))
    except ValueError:
        raise ValueError(
            f"Shape mismatch: {mode} mode expects {n_layers * params_per_layer} params.")
    wires = list(range(n_qubits))
    for l in range(n_layers):
        heft_va_layer(params[l], wires, p_noise, mode=mode)


def hea_ansatz(params, n_qubits, n_layers, p_noise=0.0):
    """Standard HEA (Paper 1, preserved verbatim)."""
    params_per_layer = n_qubits + (n_qubits - 1)
    params = params.reshape((n_layers, params_per_layer))
    wires = list(range(n_qubits))
    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RY(params[l, i], wires=wires[i])
            if p_noise > 0:
                qml.DepolarizingChannel(p_noise / 10, wires=wires[i])
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
            if p_noise > 0:
                qml.DepolarizingChannel(p_noise, wires=wires[i])

# ===========================================================================
# --- Hamiltonian Definitions (verbatim from Paper 1) ---
# ===========================================================================

def ising_hamiltonian(n_qubits: int, J: float = 1.0, h: float = 1.0,
                      periodic: bool = True) -> qml.Hamiltonian:
    """TFIM with PBC (Paper 1, verbatim)."""
    coeffs, ops = [], []
    for i in range(n_qubits):
        coeffs.append(-J)
        ops.append(qml.PauliZ(i) @ qml.PauliZ((i + 1) % n_qubits))
    for i in range(n_qubits):
        coeffs.append(-h)
        ops.append(qml.PauliX(i))
    return qml.Hamiltonian(coeffs, ops)


def heisenberg_hamiltonian(n_qubits: int, jx: float = 1.0, jy: float = 1.0,
                            jz: float = 1.0, periodic: bool = True) -> qml.Hamiltonian:
    """Heisenberg XXZ with PBC (Paper 1, verbatim)."""
    coeffs, ops = [], []
    for i in range(n_qubits):
        coeffs.extend([jx, jy, jz])
        ops.extend([
            qml.PauliX(i) @ qml.PauliX((i + 1) % n_qubits),
            qml.PauliY(i) @ qml.PauliY((i + 1) % n_qubits),
            qml.PauliZ(i) @ qml.PauliZ((i + 1) % n_qubits)
        ])
    return qml.Hamiltonian(coeffs, ops)


def get_hamiltonian(name: str, n_qubits: int) -> qml.Hamiltonian:
    """Factory (Paper 1, verbatim)."""
    key = name.lower()
    if key == 'tfim':
        return ising_hamiltonian(n_qubits)
    if key == 'heisenberg':
        return heisenberg_hamiltonian(n_qubits)
    raise ValueError(f"Unknown Hamiltonian: {name}")

# ===========================================================================
# --- Initialization Functions ---
# ===========================================================================

def heft_va_init_fn(n_qubits: int, n_layers: int,
                    kappa: float = KAPPA, mode: str = 'spin') -> np.ndarray:
    """Static H-EFT-VA initialization (Paper 1, verbatim)."""
    if mode == 'spin':
        params_per_layer = n_qubits + (n_qubits - 1)
        mu = 0
    else:
        params_per_layer = 2 * n_qubits + (n_qubits - 1)
        mu = 0.1
    n_params = n_layers * params_per_layer
    scale = kappa / (n_layers * n_qubits)
    return np.random.normal(mu, scale, size=(n_params,))


def hea_init_fn(n_qubits: int, n_layers: int) -> np.ndarray:
    """HEA initialization (Paper 1, verbatim)."""
    params_per_layer = n_qubits + (n_qubits - 1)
    return np.random.uniform(0, 2 * np.pi,
                             size=(n_layers * params_per_layer,))


# [A-H-EFT NEW]
def sigma_crit(n_qubits: int, n_layers: int,
               c2: float = SIGMA_CRIT_C2) -> float:
    """
    Critical cutoff from Theorem 1 of the paper:
        sigma_crit(N, L) = c2 / sqrt(L * N)
    Above this scale, BP avoidance is no longer guaranteed.
    """
    return c2 / np.sqrt(n_layers * n_qubits)


# [A-H-EFT NEW]
def sigma_schedule(t: int, sigma0: float,
                   lam: float = LAMBDA_DEFAULT) -> float:
    """
    Exponential growth schedule: sigma(t) = sigma0 * exp(lambda * t).
    Used only in Phase II (t >= t_switch).
    """
    return sigma0 * np.exp(lam * t)

# ===========================================================================
# --- [A-H-EFT NEW] Adaptive Optimizer ---
# ===========================================================================

def optimize_adaptive(
    qnode: Callable,
    n_qubits: int,
    n_layers: int,
    total_steps: int      = N_ADAPTIVE_STEPS,
    lr: float             = OPTIMIZER_LR,
    lam: float            = LAMBDA_DEFAULT,
    delta_switch: float   = DELTA_SWITCH_DEFAULT,
    kappa: float          = KAPPA,
    c2: float             = SIGMA_CRIT_C2,
    optimizer_name: str   = 'Adam',
    seed: int             = 0,
) -> Dict[str, Any]:
    """
    Two-phase Adaptive H-EFT-VA optimizer (Algorithm 1 of the paper).

    Phase I  (t < t_switch):
        Standard H-EFT-VA gradient descent with fixed UV-cutoff sigma0.
        Ends when ||grad C|| < delta_switch.

    Phase II (t >= t_switch):
        Controlled expansion: at each step t, a perturbation drawn from
        N(0, [sigma(t)^2 - sigma(t-1)^2] * I) is added to the warm-started
        parameters before the gradient step. sigma(t) is clamped at
        sigma_crit(N, L) to guarantee BP avoidance (Corollary 1).

    Returns a dict with:
        'history'       : energy at every step (length = total_steps)
        'sigma_history' : effective sigma at every step
        'grad_norm_history': ||grad C|| at every step
        't_switch'      : step at which Phase II began (-1 if never triggered)
        'final_params'  : optimized parameter array
    """
    np.random.seed(seed)

    # --- Setup ---
    sigma0    = kappa / (n_layers * n_qubits)
    sig_crit  = sigma_crit(n_qubits, n_layers, c2)
    n_params  = n_layers * (n_qubits + (n_qubits - 1))  # spin mode

    params = qml.numpy.array(
        np.random.normal(0, sigma0, size=(n_params,)),
        requires_grad=True
    )

    if optimizer_name == 'Adam':
        opt = qml.AdamOptimizer(stepsize=lr)
    elif optimizer_name == 'SGD':
        opt = qml.GradientDescentOptimizer(stepsize=lr)
    elif optimizer_name == 'RMSProp':
        opt = qml.RMSPropOptimizer(stepsize=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    history            = []
    sigma_history      = []
    grad_norm_history  = []
    t_switch           = -1      # -1 means Phase I never completed
    current_sigma      = sigma0

    for t in range(total_steps):

        # --- Compute gradient for logging ---
        g = qml.grad(qnode, argnum=0)(params)
        if isinstance(g, tuple):
            g = np.array(g).flatten()
        grad_norm = float(np.linalg.norm(g))
        grad_norm_history.append(grad_norm)

        # --- Phase I -> Phase II switch check ---
        if t_switch == -1 and grad_norm < delta_switch:
            t_switch = t

        # --- Phase II: apply controlled perturbation before gradient step ---
        if t_switch != -1 and t >= t_switch:
            # Compute new sigma, clamped at sigma_crit
            sigma_new = min(sigma_schedule(t - t_switch, sigma0, lam), sig_crit)
            sigma_prev = min(sigma_schedule(max(t - t_switch - 1, 0), sigma0, lam),
                             sig_crit)
            # Incremental variance for this step
            delta_var = max(sigma_new ** 2 - sigma_prev ** 2, 0.0)
            if delta_var > 0:
                xi = np.random.normal(0, np.sqrt(delta_var), size=(n_params,))
                params = qml.numpy.array(
                    np.array(params) + xi, requires_grad=True)
            current_sigma = sigma_new
        else:
            current_sigma = sigma0

        sigma_history.append(current_sigma)

        # --- Gradient step ---
        params, energy = opt.step_and_cost(qnode, params)
        history.append(float(energy))

    return {
        'history'          : history,
        'sigma_history'    : sigma_history,
        'grad_norm_history': grad_norm_history,
        't_switch'         : t_switch,
        'final_params'     : np.array(params),
    }


# ===========================================================================
# --- Utility Functions (verbatim from Paper 1) ---
# ===========================================================================

def gradient_variance_at_init(qnode: Callable, init_fn: Callable,
                               seeds: range = SEEDS) -> Tuple[float, float]:
    """Mean squared gradient norm at initialization (Paper 1, verbatim)."""
    vals = []
    for s in seeds:
        np.random.seed(s)
        p0 = qml.numpy.array(init_fn(), requires_grad=True)
        g  = qml.grad(qnode, argnum=0)(p0)
        if isinstance(g, tuple):
            g = np.array(g).flatten()
        vals.append(np.mean(g ** 2) if g.size > 0 else 0.0)
    return float(np.mean(vals)), float(np.std(vals))


def optimize_vqe(qnode: Callable, init_params: np.ndarray,
                 steps: int, lr: float,
                 optimizer_name: str = 'Adam') -> Tuple[np.ndarray, List[float]]:
    """Static VQE optimizer (Paper 1, verbatim)."""
    if optimizer_name == 'Adam':
        opt = qml.AdamOptimizer(stepsize=lr)
    elif optimizer_name == 'SGD':
        opt = qml.GradientDescentOptimizer(stepsize=lr)
    elif optimizer_name == 'RMSProp':
        opt = qml.RMSPropOptimizer(stepsize=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    params  = init_params
    history = []
    for _ in range(steps):
        params, energy = opt.step_and_cost(qnode, params)
        history.append(float(energy))
    return params, history


def entanglement_entropy(state_vector, subsystem=1):
    """Von Neumann entropy of half-chain (Paper 1, verbatim)."""
    n    = int(np.log2(state_vector.size))
    k    = subsystem
    dimA = 2 ** k
    dimB = 2 ** (n - k)
    psi  = state_vector.reshape((dimA, dimB))
    rhoA = psi.dot(psi.conj().T)
    vals = np.linalg.eigvalsh(rhoA)
    vals = vals[vals > 1e-12]
    return float(-np.sum(vals * np.log(vals)))


def expressibility_metric(ansatz_func: Callable, n_qubits: int,
                           n_layers: int, n_samples: int = 500) -> float:
    """Mean purity expressibility proxy (Paper 1, verbatim)."""
    nparams = n_layers * (n_qubits + (n_qubits - 1))
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def get_state(p):
        ansatz_func(p, n_qubits, n_layers)
        return qml.state()

    purities = []
    for _ in range(n_samples):
        p     = np.random.uniform(0, 2 * np.pi, size=nparams)
        state = get_state(p)
        purities.append(float(np.real(np.sum(np.abs(state) ** 4))))
    return float(np.mean(purities))


def get_ground_state_vector(H: qml.Hamiltonian) -> np.ndarray:
    """Exact diagonalization (Paper 1, verbatim)."""
    H_matrix = qml.matrix(H)
    _, eigenvectors = np.linalg.eigh(H_matrix)
    return eigenvectors[:, 0]


def fidelity(state_vector_1: np.ndarray,
             state_vector_2: np.ndarray) -> float:
    """F = |<psi1|psi2>|^2 (Paper 1, verbatim)."""
    return float(np.abs(np.vdot(state_vector_1, state_vector_2)) ** 2)


# [A-H-EFT NEW]
def reference_state_gap(H: qml.Hamiltonian) -> float:
    """
    Delta_ref = 1 - |<0^N | phi_0>|^2.
    Measures how far the ground state is from the computational zero state.
    Delta_ref -> 1: hard case for static H-EFT-VA.
    """
    gs = get_ground_state_vector(H)
    zero_overlap = np.abs(gs[0]) ** 2   # amplitude on |00...0>
    return float(1.0 - zero_overlap)


# [A-H-EFT NEW]
def effective_dimension(params: np.ndarray, n_qubits: int,
                        hamming_threshold: float = 1e-6) -> int:
    """
    Estimate d_eff by counting computational basis states
    whose amplitude magnitude exceeds `hamming_threshold`.
    Used to track Hilbert-space expansion during Phase II.
    """
    dev = qml.device('default.qubit', wires=n_qubits)
    n_layers = params.shape[0] // (n_qubits + (n_qubits - 1))

    @qml.qnode(dev)
    def get_state(p):
        heft_va_ansatz(p, n_qubits, n_layers)
        return qml.state()

    state = np.array(get_state(params))
    return int(np.sum(np.abs(state) > hamming_threshold))


# ===========================================================================
# --- File Handling (verbatim from Paper 1) ---
# ===========================================================================

class HamidQuantumEncoder(json.JSONEncoder):
    """Custom JSON encoder (Paper 1, verbatim)."""
    def default(self, obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)


def save_results(name: str, data: Dict[str, Any]):
    os.makedirs(RES_DIR, exist_ok=True)
    with open(os.path.join(RES_DIR, f"{name}.json"), 'w') as f:
        json.dump(data, f, indent=4, cls=HamidQuantumEncoder)


def load_results(name: str) -> Dict[str, Any]:
    with open(os.path.join(RES_DIR, f"{name}.json"), 'r') as f:
        return json.load(f)


def save_plot(fig: plt.Figure, name: str):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"),
                format='pdf', bbox_inches='tight')
    plt.close(fig)

# ===========================================================================
# ==========  ADAPTIVE BENCHMARK TESTS  =====================================
# ===========================================================================

# ---------------------------------------------------------------------------
# AT1 — Adaptive GV Scaling
# Goal: Show Var[grad C] stays in Omega(1/poly(N)) throughout both phases.
# Mirrors T1 of Paper 1 but measures GV at the END of Phase II.
# ---------------------------------------------------------------------------

def test_AT1_adaptive_gv_scaling():
    """
    [AT1] Gradient variance scaling under adaptive UV-cutoff.
    Measures GV at initialization (Phase I) and after adaptive expansion
    (Phase II) for the same (N, L) grid as Paper 1.
    """
    print("\n--- Running AT1: Adaptive GV Scaling ---")
    results = {}
    start_time = time.time()

    for n in QUBIT_LIST:
        for L in LAYER_LIST:
            dev = qml.device('default.qubit', wires=n)

            @qml.qnode(dev)
            def qnode(params):
                heft_va_ansatz(params, n, L)
                return qml.expval(get_hamiltonian(HAMILTONIAN_NAME, n))

            # Phase I GV (identical to Paper 1 T1)
            init_fn = lambda: heft_va_init_fn(n, L)
            mean_gv_p1, std_gv_p1 = gradient_variance_at_init(
                qnode, init_fn, seeds=SEEDS)

            # Phase II GV: evaluate at sigma = sigma_crit (worst-case expansion)
            sig_c = sigma_crit(n, L)
            init_fn_p2 = lambda: np.random.normal(0, sig_c,
                            size=(L * (n + (n - 1)),))
            mean_gv_p2, std_gv_p2 = gradient_variance_at_init(
                qnode, init_fn_p2, seeds=SEEDS)

            key = f"N{n}_L{L}"
            results[key] = {
                'phase1_mean_gv': mean_gv_p1,
                'phase1_std_gv' : std_gv_p1,
                'phase2_mean_gv': mean_gv_p2,
                'phase2_std_gv' : std_gv_p2,
                'sigma_crit'    : sig_c,
                'n_qubits'      : n,
                'n_layers'      : L,
            }
            print(f"  {key}: Phase-I GV={mean_gv_p1:.3e} | "
                  f"Phase-II GV={mean_gv_p2:.3e} (sigma_crit={sig_c:.4f})")

    save_results("testAT1_adaptive_gv_scaling", results)
    print(f"AT1 completed in {time.time() - start_time:.2f}s")


# ---------------------------------------------------------------------------
# AT2 — Critical Cutoff Sweep
# Goal: Empirically locate sigma_crit by sweeping sigma at N=14, L=14.
# Validates Theorem 1 of the paper.
# ---------------------------------------------------------------------------

def test_AT2_critical_cutoff_sweep():
    """
    [AT2] Critical cutoff sweep.
    Sweeps sigma from sigma0 (Paper 1 EFT scale) up to sigma=1.0
    and measures GV at each point. The BP transition should occur
    near sigma_crit(N=14, L=14) predicted by Theorem 1.
    """
    print("\n--- Running AT2: Critical Cutoff Sweep (N=14, L=14) ---")
    results  = {}
    start_time = time.time()

    N_TEST, L_TEST = 14, 14
    n_params = L_TEST * (N_TEST + (N_TEST - 1))
    sig_c    = sigma_crit(N_TEST, L_TEST)

    # Dense sweep: log-spaced from sigma0 to 1.0
    sigma0 = KAPPA / (L_TEST * N_TEST)
    sigmas = np.logspace(np.log10(sigma0), np.log10(1.0), 30)

    dev = qml.device('default.qubit', wires=N_TEST)

    @qml.qnode(dev)
    def qnode(params):
        heft_va_ansatz(params, N_TEST, L_TEST)
        return qml.expval(get_hamiltonian(HAMILTONIAN_NAME, N_TEST))

    gv_means, gv_stds = [], []
    for sig in sigmas:
        init_fn = lambda s=sig: np.random.normal(0, s, size=(n_params,))
        mean_gv, std_gv = gradient_variance_at_init(
            qnode, init_fn, seeds=SEEDS)
        gv_means.append(mean_gv)
        gv_stds.append(std_gv)
        print(f"  sigma={sig:.4e}: GV={mean_gv:.3e} ± {std_gv:.3e}")

    results.update({
        'sigmas'        : sigmas.tolist(),
        'mean_gv'       : gv_means,
        'std_gv'        : gv_stds,
        'sigma_crit'    : sig_c,
        'sigma0'        : float(sigma0),
        'n_qubits'      : N_TEST,
        'n_layers'      : L_TEST,
    })
    save_results("testAT2_critical_cutoff_sweep", results)
    print(f"AT2 completed in {time.time() - start_time:.2f}s")


# ---------------------------------------------------------------------------
# AT3 — Phase Transition: GV across Phase I / switch / Phase II
# Goal: Show GV timeline during a single adaptive run.
# Validates Corollary 1 (Safe Expansion) of the paper.
# ---------------------------------------------------------------------------

def test_AT3_phase_transition_gv():
    """
    [AT3] GV across Phase I / Phase II transition.
    Runs a full adaptive optimization at (N=8, L=8) and records
    gradient norm at every step. The vertical dashed line at t_switch
    in the plot (Fig. 1b of the paper) should show no spike in GV.
    """
    print("\n--- Running AT3: Phase Transition GV (N=8, L=8) ---")
    start_time = time.time()

    N_TEST, L_TEST = 8, 8
    dev = qml.device('default.qubit', wires=N_TEST)

    @qml.qnode(dev)
    def qnode(params):
        heft_va_ansatz(params, N_TEST, L_TEST)
        return qml.expval(get_hamiltonian(HAMILTONIAN_NAME, N_TEST))

    # Run multiple seeds for statistical reliability
    all_grad_norms = []
    all_t_switch   = []
    N_RUNS = 10

    for seed in range(N_RUNS):
        run = optimize_adaptive(
            qnode, N_TEST, L_TEST,
            total_steps=N_ADAPTIVE_STEPS,
            lr=OPTIMIZER_LR,
            lam=LAMBDA_DEFAULT,
            delta_switch=DELTA_SWITCH_DEFAULT,
            seed=seed,
        )
        all_grad_norms.append(run['grad_norm_history'])
        all_t_switch.append(run['t_switch'])
        print(f"  Seed {seed}: t_switch={run['t_switch']}, "
              f"E_final={run['history'][-1]:.4f}")

    gn_arr = np.array(all_grad_norms)
    results = {
        'mean_grad_norm' : np.mean(gn_arr, axis=0).tolist(),
        'std_grad_norm'  : np.std(gn_arr, axis=0).tolist(),
        'all_t_switch'   : all_t_switch,
        'mean_t_switch'  : float(np.mean([x for x in all_t_switch if x >= 0])) if any(x >= 0 for x in all_t_switch) else -1.0,
        'n_qubits'       : N_TEST,
        'n_layers'       : L_TEST,
        'total_steps'    : N_ADAPTIVE_STEPS,
    }
    save_results("testAT3_phase_transition_gv", results)
    print(f"AT3 completed in {time.time() - start_time:.2f}s")


# ---------------------------------------------------------------------------
# AT4 — Adaptive Convergence vs Baselines
# Goal: Show A-H-EFT reaches deeper energy minima than static H-EFT-VA and HEA.
# Mirrors T5 of Paper 1 but adds adaptive run.
# ---------------------------------------------------------------------------

def test_AT4_adaptive_convergence():
    """
    [AT4] Noiseless convergence: A-H-EFT vs static H-EFT-VA vs HEA.
    Uses the same (N, L) subset as Paper 1 T5: N in {4, 8, 12},
    L in {4, 8, 12}. A-H-EFT uses N_ADAPTIVE_STEPS; others use
    the same budget for fair comparison.
    """
    print("\n--- Running AT4: Adaptive Convergence vs Baselines ---")
    results = {}
    start_time = time.time()

    QUBIT_SUBSET = [4, 8, 12]
    LAYER_SUBSET = [4, 8, 12]

    for n in QUBIT_SUBSET:
        for L in LAYER_SUBSET:
            dev = qml.device('default.qubit', wires=n)

            @qml.qnode(dev)
            def qnode(params):
                heft_va_ansatz(params, n, L)
                return qml.expval(get_hamiltonian(HAMILTONIAN_NAME, n))

            @qml.qnode(dev)
            def hea_qnode(params):
                hea_ansatz(params, n, L)
                return qml.expval(get_hamiltonian(HAMILTONIAN_NAME, n))

            # --- A-H-EFT (adaptive, seed=0) ---
            run_aheft = optimize_adaptive(
                qnode, n, L,
                total_steps=N_ADAPTIVE_STEPS,
                lr=OPTIMIZER_LR,
                lam=LAMBDA_DEFAULT,
                delta_switch=DELTA_SWITCH_DEFAULT,
                seed=0,
            )

            # --- Static H-EFT-VA (same total steps) ---
            init_static = heft_va_init_fn(n, L)
            _, hist_static = optimize_vqe(
                qnode, init_static,
                steps=N_ADAPTIVE_STEPS,
                lr=OPTIMIZER_LR, optimizer_name='Adam')

            # --- HEA ---
            init_hea = hea_init_fn(n, L)
            _, hist_hea = optimize_vqe(
                hea_qnode, init_hea,
                steps=N_ADAPTIVE_STEPS,
                lr=OPTIMIZER_LR, optimizer_name='Adam')

            key = f"N{n}_L{L}"
            results[key] = {
                'aheft_history' : run_aheft['history'],
                'static_history': hist_static,
                'hea_history'   : hist_hea,
                'aheft_t_switch': run_aheft['t_switch'],
                'n_qubits'      : n,
                'n_layers'      : L,
            }
            print(f"  {key}: A-H-EFT={run_aheft['history'][-1]:.4f} | "
                  f"Static={hist_static[-1]:.4f} | HEA={hist_hea[-1]:.4f} | "
                  f"t_switch={run_aheft['t_switch']}")

    save_results("testAT4_adaptive_convergence", results)
    print(f"AT4 completed in {time.time() - start_time:.2f}s")


# ---------------------------------------------------------------------------
# AT5 — Convergence vs System Size (Adaptive)
# Goal: Show performance gap widens with N.
# Mirrors T6 of Paper 1 with adaptive runner added.
# ---------------------------------------------------------------------------

def test_AT5_convergence_vs_system_size():
    """
    [AT5] Final energy vs N for A-H-EFT, static H-EFT-VA, and HEA.
    Fixed L=2 (depth-limited regime) to match Paper 1 T6.
    """
    print("\n--- Running AT5: Convergence vs System Size ---")
    results = {}
    start_time = time.time()
    L_TEST = 2

    for n in QUBIT_LIST:
        dev = qml.device('default.qubit', wires=n)

        @qml.qnode(dev)
        def qnode(params):
            heft_va_ansatz(params, n, L_TEST)
            return qml.expval(get_hamiltonian(HAMILTONIAN_NAME, n))

        @qml.qnode(dev)
        def hea_qnode(params):
            hea_ansatz(params, n, L_TEST)
            return qml.expval(get_hamiltonian(HAMILTONIAN_NAME, n))

        # A-H-EFT
        run_aheft = optimize_adaptive(
            qnode, n, L_TEST,
            total_steps=N_ADAPTIVE_STEPS,
            lr=OPTIMIZER_LR,
            lam=LAMBDA_DEFAULT,
            delta_switch=DELTA_SWITCH_DEFAULT,
            seed=0,
        )

        # Static H-EFT-VA
        _, hist_static = optimize_vqe(
            qnode, heft_va_init_fn(n, L_TEST),
            steps=N_ADAPTIVE_STEPS, lr=OPTIMIZER_LR)

        # HEA
        _, hist_hea = optimize_vqe(
            hea_qnode, hea_init_fn(n, L_TEST),
            steps=N_ADAPTIVE_STEPS, lr=OPTIMIZER_LR)

        key = f"N{n}_L{L_TEST}"
        results[key] = {
            'aheft_final' : run_aheft['history'][-1],
            'static_final': hist_static[-1],
            'hea_final'   : hist_hea[-1],
            'n_qubits'    : n,
            'n_layers'    : L_TEST,
        }
        print(f"  {key}: A-H-EFT={run_aheft['history'][-1]:.4f} | "
              f"Static={hist_static[-1]:.4f} | HEA={hist_hea[-1]:.4f}")

    save_results("testAT5_convergence_vs_system_size", results)
    print(f"AT5 completed in {time.time() - start_time:.2f}s")


# ---------------------------------------------------------------------------
# AT6 — Ground-State Fidelity
# Goal: Show A-H-EFT bridges the reference-state gap.
# Mirrors T16 of Paper 1 with adaptive runner and Delta_ref annotation.
# ---------------------------------------------------------------------------

def test_AT6_ground_state_fidelity():
    """
    [AT6] Ground-state fidelity: A-H-EFT vs static H-EFT-VA vs HEA.
    N=6, L swept over LAYER_LIST, 5 seeds (matching Paper 1 T16).
    """
    print("\n--- Running AT6: Ground-State Fidelity ---")
    results = {}
    start_time = time.time()

    N_TEST       = 6
    N_STAT_SEEDS = 5
    dev          = qml.device('default.qubit', wires=N_TEST)
    H            = get_hamiltonian(HAMILTONIAN_NAME, N_TEST)
    gs_vec       = get_ground_state_vector(H)
    delta_ref    = reference_state_gap(H)

    aheft_f_means,  aheft_f_stds  = [], []
    static_f_means, static_f_stds = [], []
    hea_f_means,    hea_f_stds    = [], []

    for L in LAYER_LIST:
        f_aheft, f_static, f_hea = [], [], []

        @qml.qnode(dev)
        def cost_node(p):
            heft_va_ansatz(p, N_TEST, L)
            return qml.expval(H)

        @qml.qnode(dev)
        def state_node(p):
            heft_va_ansatz(p, N_TEST, L)
            return qml.state()

        @qml.qnode(dev)
        def hea_cost(p):
            hea_ansatz(p, N_TEST, L)
            return qml.expval(H)

        @qml.qnode(dev)
        def hea_state(p):
            hea_ansatz(p, N_TEST, L)
            return qml.state()

        for s in range(N_STAT_SEEDS):
            # A-H-EFT
            run_a = optimize_adaptive(
                cost_node, N_TEST, L,
                total_steps=N_ADAPTIVE_STEPS,
                lr=OPTIMIZER_LR,
                lam=LAMBDA_DEFAULT,
                delta_switch=DELTA_SWITCH_DEFAULT,
                seed=s,
            )
            f_aheft.append(fidelity(gs_vec, state_node(run_a['final_params'])))

            # Static H-EFT-VA
            p_s, _ = optimize_vqe(cost_node, heft_va_init_fn(N_TEST, L),
                                   steps=N_ADAPTIVE_STEPS, lr=OPTIMIZER_LR)
            f_static.append(fidelity(gs_vec, state_node(p_s)))

            # HEA
            p_h, _ = optimize_vqe(hea_cost, hea_init_fn(N_TEST, L),
                                   steps=N_ADAPTIVE_STEPS, lr=OPTIMIZER_LR)
            f_hea.append(fidelity(gs_vec, hea_state(p_h)))

        aheft_f_means.append(float(np.mean(f_aheft)))
        aheft_f_stds.append(float(np.std(f_aheft)))
        static_f_means.append(float(np.mean(f_static)))
        static_f_stds.append(float(np.std(f_static)))
        hea_f_means.append(float(np.mean(f_hea)))
        hea_f_stds.append(float(np.std(f_hea)))

        print(f"  L={L}: A-H-EFT={aheft_f_means[-1]:.4f}±{aheft_f_stds[-1]:.4f} | "
              f"Static={static_f_means[-1]:.4f} | HEA={hea_f_means[-1]:.4f}")

    results.update({
        'aheft_fid_mean' : aheft_f_means,
        'aheft_fid_std'  : aheft_f_stds,
        'static_fid_mean': static_f_means,
        'static_fid_std' : static_f_stds,
        'hea_fid_mean'   : hea_f_means,
        'hea_fid_std'    : hea_f_stds,
        'layer_list'     : LAYER_LIST,
        'n_qubits'       : N_TEST,
        'delta_ref'      : delta_ref,
    })
    save_results("testAT6_ground_state_fidelity", results)
    print(f"AT6 completed in {time.time() - start_time:.2f}s "
          f"(Delta_ref={delta_ref:.4f})")


# ---------------------------------------------------------------------------
# AT7 — Reference-State Gap Delta_ref vs N
# Goal: Show Delta_ref grows with N, motivating the adaptive strategy.
# ---------------------------------------------------------------------------

def test_AT7_reference_state_gap():
    """
    [AT7] Delta_ref vs N for TFIM and Heisenberg XXZ.
    No circuit needed — pure Hamiltonian property.
    """
    print("\n--- Running AT7: Reference-State Gap vs N ---")
    results = {}
    start_time = time.time()

    tfim_gaps, heis_gaps = [], []

    for n in QUBIT_LIST:
        H_tfim = get_hamiltonian('tfim', n)
        H_heis = get_hamiltonian('heisenberg', n)
        dr_tfim = reference_state_gap(H_tfim)
        dr_heis = reference_state_gap(H_heis)
        tfim_gaps.append(dr_tfim)
        heis_gaps.append(dr_heis)
        print(f"  N={n}: Delta_ref(TFIM)={dr_tfim:.4f} | "
              f"Delta_ref(Heisenberg)={dr_heis:.4f}")

    results.update({
        'qubit_list'  : QUBIT_LIST,
        'tfim_gaps'   : tfim_gaps,
        'heis_gaps'   : heis_gaps,
    })
    save_results("testAT7_reference_state_gap", results)
    print(f"AT7 completed in {time.time() - start_time:.2f}s")


# ---------------------------------------------------------------------------
# AT8 — Effective Hilbert Space Growth
# Goal: Show d_eff expands monotonically and polynomially during Phase II.
# Validates Lemma 1 (Monotone Growth Lemma) of the paper.
# ---------------------------------------------------------------------------

def test_AT8_effective_dimension_growth():
    """
    [AT8] Track d_eff throughout an adaptive run at (N=8, L=8).
    Records d_eff at every 10th optimization step.
    """
    print("\n--- Running AT8: Effective Dimension Growth ---")
    start_time = time.time()

    N_TEST, L_TEST  = 8, 8
    SAMPLE_EVERY    = 5    # record d_eff every N steps to keep runtime feasible
    dev = qml.device('default.qubit', wires=N_TEST)

    @qml.qnode(dev)
    def qnode(params):
        heft_va_ansatz(params, N_TEST, L_TEST)
        return qml.expval(get_hamiltonian(HAMILTONIAN_NAME, N_TEST))

    # We re-implement adaptive loop here with d_eff logging
    sigma0   = KAPPA / (L_TEST * N_TEST)
    sig_c    = sigma_crit(N_TEST, L_TEST)
    n_params = L_TEST * (N_TEST + (N_TEST - 1))

    np.random.seed(0)
    params  = qml.numpy.array(np.random.normal(0, sigma0, size=(n_params,)),
                               requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=OPTIMIZER_LR)

    deff_list, step_list, sigma_list, t_switch = [], [], [], -1
    current_sigma = sigma0

    for t in range(N_ADAPTIVE_STEPS):
        g = qml.grad(qnode, argnum=0)(params)
        if isinstance(g, tuple):
            g = np.array(g).flatten()
        grad_norm = float(np.linalg.norm(g))

        if t_switch == -1 and grad_norm < DELTA_SWITCH_DEFAULT:
            t_switch = t

        if t_switch != -1 and t >= t_switch:
            sigma_new  = min(sigma_schedule(t - t_switch, sigma0, LAMBDA_DEFAULT), sig_c)
            sigma_prev = min(sigma_schedule(max(t - t_switch - 1, 0), sigma0, LAMBDA_DEFAULT), sig_c)
            delta_var  = max(sigma_new ** 2 - sigma_prev ** 2, 0.0)
            if delta_var > 0:
                xi = np.random.normal(0, np.sqrt(delta_var), size=(n_params,))
                params = qml.numpy.array(np.array(params) + xi, requires_grad=True)
            current_sigma = sigma_new
        else:
            current_sigma = sigma0

        params, _ = opt.step_and_cost(qnode, params)

        if t % SAMPLE_EVERY == 0:
            deff = effective_dimension(np.array(params), N_TEST)
            deff_list.append(deff)
            step_list.append(t)
            sigma_list.append(current_sigma)
            print(f"  t={t}: d_eff={deff} | sigma={current_sigma:.5f} | "
                  f"phase={'II' if t_switch != -1 and t >= t_switch else 'I'}")

    results = {
        'steps'       : step_list,
        'deff'        : deff_list,
        'sigma'       : sigma_list,
        't_switch'    : t_switch,
        'sigma_crit'  : sig_c,
        'n_qubits'    : N_TEST,
        'n_layers'    : L_TEST,
    }
    save_results("testAT8_effective_dimension_growth", results)
    print(f"AT8 completed in {time.time() - start_time:.2f}s")


# ---------------------------------------------------------------------------
# AT9 — Entanglement Entropy: Phase I vs Phase II
# Goal: Show entanglement grows further in Phase II.
# Mirrors T13 of Paper 1 with Phase I / Phase II separation.
# ---------------------------------------------------------------------------

def test_AT9_entanglement_growth():
    """
    [AT9] Von Neumann entropy vs circuit depth for A-H-EFT, static, and HEA.
    Architecture capacity: parameters sampled uniformly from [0, 2pi].
    N=8, averaged over N_SAMPLES random parameter draws per L.
    """
    print("\n--- Running AT9: Entanglement Growth ---")
    results = {}
    start_time = time.time()

    N_TEST    = 8
    N_SAMPLES = 15
    dev       = qml.device('default.qubit', wires=N_TEST)

    aheft_means, aheft_stds   = [], []
    static_means, static_stds = [], []
    hea_means, hea_stds       = [], []

    for L in LAYER_LIST:
        n_params = L * (N_TEST + (N_TEST - 1))
        a_vals, s_vals, h_vals = [], [], []

        for _ in range(N_SAMPLES):
            # A-H-EFT: sample at sigma_crit to represent Phase II capacity
            sig_c = sigma_crit(N_TEST, L)
            p_a   = np.random.normal(0, sig_c, size=(n_params,))

            @qml.qnode(dev)
            def state_a(p):
                heft_va_ansatz(p, N_TEST, L)
                return qml.state()

            a_vals.append(entanglement_entropy(
                state_a(p_a), subsystem=N_TEST // 2))

            # Static H-EFT-VA: sample at sigma0
            sigma0 = KAPPA / (L * N_TEST)
            p_s    = np.random.normal(0, sigma0, size=(n_params,))
            s_vals.append(entanglement_entropy(
                state_a(p_s), subsystem=N_TEST // 2))

            # HEA: uniform [0, 2pi]
            p_h = np.random.uniform(0, 2 * np.pi, size=(n_params,))

            @qml.qnode(dev)
            def state_h(p):
                hea_ansatz(p, N_TEST, L)
                return qml.state()

            h_vals.append(entanglement_entropy(
                state_h(p_h), subsystem=N_TEST // 2))

        aheft_means.append(float(np.mean(a_vals)))
        aheft_stds.append(float(np.std(a_vals)))
        static_means.append(float(np.mean(s_vals)))
        static_stds.append(float(np.std(s_vals)))
        hea_means.append(float(np.mean(h_vals)))
        hea_stds.append(float(np.std(h_vals)))
        print(f"  L={L}: A-H-EFT Sv={aheft_means[-1]:.3f} | "
              f"Static Sv={static_means[-1]:.3f} | HEA Sv={hea_means[-1]:.3f}")

    results.update({
        'aheft_entropy' : aheft_means,  'aheft_std' : aheft_stds,
        'static_entropy': static_means, 'static_std': static_stds,
        'hea_entropy'   : hea_means,    'hea_std'   : hea_stds,
        'n_qubits'      : N_TEST,       'layers'    : LAYER_LIST,
    })
    save_results("testAT9_entanglement_growth", results)
    print(f"AT9 completed in {time.time() - start_time:.2f}s")


# ---------------------------------------------------------------------------
# AT10 — Expressibility Proxy
# Goal: Show A-H-EFT approaches Haar limit more closely than static.
# Mirrors T14 of Paper 1.
# ---------------------------------------------------------------------------

def test_AT10_expressibility():
    """
    [AT10] Mean purity expressibility proxy: A-H-EFT vs static vs HEA vs Haar.
    N=6, 500 samples per data point (Paper 1 standard).
    """
    print("\n--- Running AT10: Expressibility Proxy ---")
    results = {}
    start_time = time.time()

    N_TEST = 6

    # We create a wrapper that samples from sigma_crit distribution
    # to represent the maximally expanded Phase II capacity
    def aheft_expanded_ansatz(p, n_qubits, n_layers):
        """Wrapper: samples at sigma_crit to test Phase-II expressibility."""
        sig_c  = sigma_crit(n_qubits, n_layers)
        n_pars = n_layers * (n_qubits + (n_qubits - 1))
        p_crit = np.random.normal(0, sig_c, size=(n_pars,))
        heft_va_ansatz(p_crit, n_qubits, n_layers)

    aheft_purity, static_purity, hea_purity = [], [], []
    haar_limit = 2 / (2 ** N_TEST + 1)

    for L in LAYER_LIST:
        p_a = expressibility_metric(
            lambda p, n, l: heft_va_ansatz(p, n, l),
            N_TEST, L, n_samples=500)
        # Static uses sigma_0 — sample params directly at EFT scale
        sigma0  = KAPPA / (L * N_TEST)
        n_pars  = L * (N_TEST + (N_TEST - 1))
        dev     = qml.device('default.qubit', wires=N_TEST)

        @qml.qnode(dev)
        def get_state_crit(dummy):
            sig_c = sigma_crit(N_TEST, L)
            p_c   = np.random.normal(0, sig_c, size=(n_pars,))
            heft_va_ansatz(p_c, N_TEST, L)
            return qml.state()

        @qml.qnode(dev)
        def get_state_s0(dummy):
            p_s0 = np.random.normal(0, sigma0, size=(n_pars,))
            heft_va_ansatz(p_s0, N_TEST, L)
            return qml.state()

        purities_a, purities_s = [], []
        for _ in range(500):
            st_a = get_state_crit(None)
            st_s = get_state_s0(None)
            purities_a.append(float(np.real(np.sum(np.abs(st_a) ** 4))))
            purities_s.append(float(np.real(np.sum(np.abs(st_s) ** 4))))

        p_a_val = float(np.mean(purities_a))
        p_s_val = float(np.mean(purities_s))
        p_h_val = expressibility_metric(hea_ansatz, N_TEST, L, n_samples=500)

        aheft_purity.append(p_a_val)
        static_purity.append(p_s_val)
        hea_purity.append(p_h_val)
        print(f"  L={L}: A-H-EFT={p_a_val:.4f} | Static={p_s_val:.4f} | "
              f"HEA={p_h_val:.4f} | Haar={haar_limit:.4f}")

    results.update({
        'aheft_purity' : aheft_purity,
        'static_purity': static_purity,
        'hea_purity'   : hea_purity,
        'haar_limit'   : haar_limit,
        'layer_list'   : LAYER_LIST,
        'n_qubits'     : N_TEST,
    })
    save_results("testAT10_expressibility", results)
    print(f"AT10 completed in {time.time() - start_time:.2f}s")


# ---------------------------------------------------------------------------
# AT11 — Noise Robustness of Adaptive Training
# Mirrors T9 of Paper 1 but with adaptive optimizer.
# ---------------------------------------------------------------------------

def test_AT11_noise_robustness():
    """
    [AT11] Adaptive training under depolarizing noise.
    N=8, L=8. p in {0, 1e-4, 1e-3, 1e-2}.
    """
    print("\n--- Running AT11: Noise Robustness (Adaptive) ---")
    results = {}
    start_time = time.time()

    N_TEST, L_TEST  = 8, 8
    P_NOISE_LIST    = [0.0, 1e-4, 1e-3, 1e-2]

    for p_noise in P_NOISE_LIST:
        dev = qml.device('default.mixed', wires=N_TEST)

        @qml.qnode(dev)
        def noisy_qnode(params):
            heft_va_ansatz(params, N_TEST, L_TEST, p_noise=p_noise)
            return qml.expval(get_hamiltonian(HAMILTONIAN_NAME, N_TEST))

        run = optimize_adaptive(
            noisy_qnode, N_TEST, L_TEST,
            total_steps=N_ADAPTIVE_STEPS,
            lr=OPTIMIZER_LR,
            lam=LAMBDA_DEFAULT,
            delta_switch=DELTA_SWITCH_DEFAULT,
            seed=0,
        )
        key = f"P_NOISE_{p_noise}"
        results[key] = {
            'history' : run['history'],
            't_switch': run['t_switch'],
        }
        print(f"  p_noise={p_noise}: E_final={run['history'][-1]:.4f} | "
              f"t_switch={run['t_switch']}")

    results.update({'n_qubits': N_TEST, 'n_layers': L_TEST})
    save_results("testAT11_noise_robustness", results)
    print(f"AT11 completed in {time.time() - start_time:.2f}s")


# ---------------------------------------------------------------------------
# AT12 — Finite-Shot Gradient Estimator under Adaptive Schedule
# Mirrors T10 of Paper 1 with adaptive runner.
# ---------------------------------------------------------------------------

def test_AT12_finite_shot_estimator():
    """
    [AT12] Finite-shot gradient MSE for A-H-EFT vs HEA.
    N=8, L=4, shots in {1000, 5000, 10000}.
    """
    print("\n--- Running AT12: Finite-Shot Gradient Estimator ---")
    results = {}
    start_time = time.time()

    N_TEST, L_TEST = 8, 4
    SHOTS_LIST     = [1000, 5000, 10000]
    N_REPS         = 20
    N_SEEDS        = 10
    n_params_heft  = L_TEST * (N_TEST + (N_TEST - 1))

    dev_exact = qml.device('default.qubit', wires=N_TEST, shots=None)

    @qml.qnode(dev_exact)
    def heft_exact(params):
        heft_va_ansatz(params, N_TEST, L_TEST)
        return qml.expval(get_hamiltonian(HAMILTONIAN_NAME, N_TEST))

    @qml.qnode(dev_exact)
    def hea_exact(params):
        hea_ansatz(params, N_TEST, L_TEST)
        return qml.expval(get_hamiltonian(HAMILTONIAN_NAME, N_TEST))

    for shots in SHOTS_LIST:
        dev_shots = qml.device('default.qubit', wires=N_TEST, shots=shots)

        @qml.qnode(dev_shots, diff_method="parameter-shift")
        def heft_shots(params):
            heft_va_ansatz(params, N_TEST, L_TEST)
            return qml.expval(get_hamiltonian(HAMILTONIAN_NAME, N_TEST))

        @qml.qnode(dev_shots, diff_method="parameter-shift")
        def hea_shots(params):
            hea_ansatz(params, N_TEST, L_TEST)
            return qml.expval(get_hamiltonian(HAMILTONIAN_NAME, N_TEST))

        heft_mse_vals, hea_mse_vals = [], []

        for s in range(N_SEEDS):
            np.random.seed(s)
            # A-H-EFT: evaluate at sigma_crit to test Phase-II shot efficiency
            sig_c = sigma_crit(N_TEST, L_TEST)
            p_a   = qml.numpy.array(
                np.random.normal(0, sig_c, size=(n_params_heft,)),
                requires_grad=True)
            p_h   = qml.numpy.array(
                hea_init_fn(N_TEST, L_TEST), requires_grad=True)

            g_exact_a = np.array(qml.grad(heft_exact)(p_a)).flatten()
            g_exact_h = np.array(qml.grad(hea_exact)(p_h)).flatten()

            g_reps_a, g_reps_h = [], []
            for _ in range(N_REPS):
                g_reps_a.append(np.array(qml.grad(heft_shots)(p_a)).flatten())
                g_reps_h.append(np.array(qml.grad(hea_shots)(p_h)).flatten())

            g_reps_a = np.array(g_reps_a)
            g_reps_h = np.array(g_reps_h)

            mse_a = float(np.mean((np.mean(g_reps_a, axis=0) - g_exact_a) ** 2)
                          + np.mean(np.var(g_reps_a, axis=0)))
            mse_h = float(np.mean((np.mean(g_reps_h, axis=0) - g_exact_h) ** 2)
                          + np.mean(np.var(g_reps_h, axis=0)))
            heft_mse_vals.append(mse_a)
            hea_mse_vals.append(mse_h)

        key = f"SHOTS_{shots}"
        results[key] = {
            'aheft_mse': float(np.mean(heft_mse_vals)),
            'hea_mse'  : float(np.mean(hea_mse_vals)),
        }
        print(f"  {key}: A-H-EFT MSE={results[key]['aheft_mse']:.3e} | "
              f"HEA MSE={results[key]['hea_mse']:.3e}")

    results.update({'n_qubits': N_TEST, 'n_layers': L_TEST})
    save_results("testAT12_finite_shot_estimator", results)
    print(f"AT12 completed in {time.time() - start_time:.2f}s")


# ---------------------------------------------------------------------------
# AT13 — Switch Criterion Sensitivity
# Goal: Characterize how delta_switch affects final fidelity.
# ---------------------------------------------------------------------------

def test_AT13_switch_sensitivity():
    """
    [AT13] Sweep delta_switch in {1e-4, 5e-4, 1e-3, 5e-3, 1e-2}.
    Measure final energy and t_switch for each value at (N=8, L=8).
    """
    print("\n--- Running AT13: Switch Criterion Sensitivity ---")
    results = {}
    start_time = time.time()

    N_TEST, L_TEST   = 8, 8
    DELTA_LIST       = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    N_SEEDS          = 10
    dev = qml.device('default.qubit', wires=N_TEST)

    @qml.qnode(dev)
    def qnode(params):
        heft_va_ansatz(params, N_TEST, L_TEST)
        return qml.expval(get_hamiltonian(HAMILTONIAN_NAME, N_TEST))

    for ds in DELTA_LIST:
        e_finals, t_switches = [], []
        for seed in range(N_SEEDS):
            run = optimize_adaptive(
                qnode, N_TEST, L_TEST,
                total_steps=N_ADAPTIVE_STEPS,
                lr=OPTIMIZER_LR,
                lam=LAMBDA_DEFAULT,
                delta_switch=ds,
                seed=seed,
            )
            e_finals.append(run['history'][-1])
            t_switches.append(run['t_switch'])

        key = f"DS_{ds}"
        results[key] = {
            'delta_switch'    : ds,
            'mean_e_final'    : float(np.mean(e_finals)),
            'std_e_final'     : float(np.std(e_finals)),
            'mean_t_switch'   : float(np.mean([x for x in t_switches if x >= 0])) if any(x >= 0 for x in t_switches) else -1.0,
            'frac_triggered'  : sum(1 for x in t_switches if x >= 0) / N_SEEDS,
        }
        print(f"  delta_switch={ds:.1e}: E_final={results[key]['mean_e_final']:.4f} | "
              f"mean t_switch={results[key]['mean_t_switch']:.1f} | "
              f"triggered={results[key]['frac_triggered']*100:.0f}%")

    results.update({'n_qubits': N_TEST, 'n_layers': L_TEST})
    save_results("testAT13_switch_sensitivity", results)
    print(f"AT13 completed in {time.time() - start_time:.2f}s")


# ---------------------------------------------------------------------------
# AT14 — Growth Constant Sensitivity (lambda sweep)
# Goal: Characterize how lambda affects convergence and BP safety.
# ---------------------------------------------------------------------------

def test_AT14_lambda_sensitivity():
    """
    [AT14] Sweep lambda in {0.005, 0.01, 0.02, 0.05, 0.1}.
    Measure final energy, sigma at end of run, and t_switch.
    At (N=8, L=8).
    """
    print("\n--- Running AT14: Growth Constant Sensitivity ---")
    results = {}
    start_time = time.time()

    N_TEST, L_TEST = 8, 8
    LAMBDA_LIST    = [0.005, 0.01, 0.02, 0.05, 0.1]
    N_SEEDS        = 10
    sig_c          = sigma_crit(N_TEST, L_TEST)
    dev = qml.device('default.qubit', wires=N_TEST)

    @qml.qnode(dev)
    def qnode(params):
        heft_va_ansatz(params, N_TEST, L_TEST)
        return qml.expval(get_hamiltonian(HAMILTONIAN_NAME, N_TEST))

    for lam in LAMBDA_LIST:
        e_finals, final_sigmas = [], []
        for seed in range(N_SEEDS):
            run = optimize_adaptive(
                qnode, N_TEST, L_TEST,
                total_steps=N_ADAPTIVE_STEPS,
                lr=OPTIMIZER_LR,
                lam=lam,
                delta_switch=DELTA_SWITCH_DEFAULT,
                seed=seed,
            )
            e_finals.append(run['history'][-1])
            final_sigmas.append(run['sigma_history'][-1])

        key = f"LAM_{lam}"
        results[key] = {
            'lambda'         : lam,
            'mean_e_final'   : float(np.mean(e_finals)),
            'std_e_final'    : float(np.std(e_finals)),
            'mean_sigma_final': float(np.mean(final_sigmas)),
            'clamped'        : float(np.mean(final_sigmas)) >= sig_c * 0.99,
        }
        print(f"  lambda={lam:.3f}: E_final={results[key]['mean_e_final']:.4f} | "
              f"sigma_final={results[key]['mean_sigma_final']:.5f} | "
              f"clamped={results[key]['clamped']}")

    results.update({'n_qubits': N_TEST, 'n_layers': L_TEST,
                    'sigma_crit': sig_c})
    save_results("testAT14_lambda_sensitivity", results)
    print(f"AT14 completed in {time.time() - start_time:.2f}s")


# ---------------------------------------------------------------------------
# AT15 — Statistical Significance: Adaptive vs All Baselines
# Mirrors T15 of Paper 1 with adaptive runner included.
# ---------------------------------------------------------------------------

def test_AT15_statistical_significance():
    """
    [AT15] Welch's t-test and Cohen's d: A-H-EFT vs static H-EFT-VA vs HEA.
    50 independent seeds, (N, L) pairs in {(4,4), (8,8), (12,12)}.
    """
    print("\n--- Running AT15: Statistical Significance ---")
    results = {}
    start_time = time.time()

    TEST_PAIRS   = [(4, 4), (8, 8), (12, 12)]
    N_STAT_RUNS  = 50

    for n, L in TEST_PAIRS:
        dev = qml.device('default.qubit', wires=n)

        @qml.qnode(dev)
        def qnode(params):
            heft_va_ansatz(params, n, L)
            return qml.expval(get_hamiltonian(HAMILTONIAN_NAME, n))

        @qml.qnode(dev)
        def hea_qnode(params):
            hea_ansatz(params, n, L)
            return qml.expval(get_hamiltonian(HAMILTONIAN_NAME, n))

        e_aheft, e_static, e_hea = [], [], []

        for s in range(N_STAT_RUNS):
            np.random.seed(s)

            # A-H-EFT
            run_a = optimize_adaptive(
                qnode, n, L,
                total_steps=N_ADAPTIVE_STEPS,
                lr=OPTIMIZER_LR,
                lam=LAMBDA_DEFAULT,
                delta_switch=DELTA_SWITCH_DEFAULT,
                seed=s,
            )
            e_aheft.append(run_a['history'][-1])

            # Static H-EFT-VA
            _, h_s = optimize_vqe(qnode, heft_va_init_fn(n, L),
                                   steps=N_ADAPTIVE_STEPS, lr=OPTIMIZER_LR)
            e_static.append(h_s[-1])

            # HEA
            _, h_h = optimize_vqe(hea_qnode, hea_init_fn(n, L),
                                   steps=N_ADAPTIVE_STEPS, lr=OPTIMIZER_LR)
            e_hea.append(h_h[-1])

        e_aheft  = np.array(e_aheft)
        e_static = np.array(e_static)
        e_hea    = np.array(e_hea)

        # Welch t-test and Cohen's d: adaptive vs static
        t_as, p_as = stats.ttest_ind(e_aheft, e_static, equal_var=False)
        d_as = (np.mean(e_aheft) - np.mean(e_static)) / np.sqrt(
            (np.std(e_aheft, ddof=1) ** 2 + np.std(e_static, ddof=1) ** 2) / 2)

        # adaptive vs HEA
        t_ah, p_ah = stats.ttest_ind(e_aheft, e_hea, equal_var=False)
        d_ah = (np.mean(e_aheft) - np.mean(e_hea)) / np.sqrt(
            (np.std(e_aheft, ddof=1) ** 2 + np.std(e_hea, ddof=1) ** 2) / 2)

        key = f"N{n}_L{L}"
        results[key] = {
            'aheft_mean' : float(np.mean(e_aheft)),
            'aheft_std'  : float(np.std(e_aheft)),
            'static_mean': float(np.mean(e_static)),
            'static_std' : float(np.std(e_static)),
            'hea_mean'   : float(np.mean(e_hea)),
            'hea_std'    : float(np.std(e_hea)),
            'p_aheft_vs_static': float(p_as),
            'cohen_d_vs_static': float(d_as),
            'p_aheft_vs_hea'   : float(p_ah),
            'cohen_d_vs_hea'   : float(d_ah),
            'n_qubits'   : n,
            'n_layers'   : L,
        }
        print(f"  {key}: A-H-EFT={results[key]['aheft_mean']:.4f} | "
              f"p(vs static)={p_as:.2e} | p(vs HEA)={p_ah:.2e}")

    save_results("testAT15_statistical_significance", results)
    print(f"AT15 completed in {time.time() - start_time:.2f}s")


# ---------------------------------------------------------------------------
# AT16 — Heisenberg XXZ Full Adaptive Benchmark
# Goal: Show model-independence on a second Hamiltonian.
# ---------------------------------------------------------------------------

def test_AT16_heisenberg_benchmark():
    """
    [AT16] Full adaptive benchmark on the Heisenberg XXZ chain.
    Covers GV scaling, convergence, and fidelity.
    Mirrors T12 of Paper 1 with adaptive runner.
    """
    print("\n--- Running AT16: Heisenberg XXZ Benchmark ---")
    results = {}
    start_time = time.time()

    # GV scaling
    for n in QUBIT_LIST:
        for L in LAYER_LIST:
            dev = qml.device('default.qubit', wires=n)

            @qml.qnode(dev)
            def qnode(params):
                heft_va_ansatz(params, n, L)
                return qml.expval(get_hamiltonian('heisenberg', n))

            # Phase I GV
            init_fn = lambda: heft_va_init_fn(n, L)
            mean_gv_p1, _ = gradient_variance_at_init(
                qnode, init_fn, seeds=range(20))

            # Phase II GV at sigma_crit
            sig_c = sigma_crit(n, L)
            init_fn_p2 = lambda s=sig_c: np.random.normal(
                0, s, size=(L * (n + (n - 1)),))
            mean_gv_p2, _ = gradient_variance_at_init(
                qnode, init_fn_p2, seeds=range(20))

            key = f"N{n}_L{L}"
            results[key] = {
                'phase1_gv': mean_gv_p1,
                'phase2_gv': mean_gv_p2,
                'n_qubits' : n,
                'n_layers' : L,
            }
            print(f"  {key}: Phase-I GV={mean_gv_p1:.3e} | "
                  f"Phase-II GV={mean_gv_p2:.3e}")

    # Convergence at representative sizes
    QUBIT_SUBSET = [4, 8, 12]
    LAYER_SUBSET = [4, 8, 12]
    for n in QUBIT_SUBSET:
        for L in LAYER_SUBSET:
            dev = qml.device('default.qubit', wires=n)

            @qml.qnode(dev)
            def qnode_c(params):
                heft_va_ansatz(params, n, L)
                return qml.expval(get_hamiltonian('heisenberg', n))

            run_a = optimize_adaptive(
                qnode_c, n, L,
                total_steps=N_ADAPTIVE_STEPS,
                lr=OPTIMIZER_LR,
                lam=LAMBDA_DEFAULT,
                delta_switch=DELTA_SWITCH_DEFAULT,
                seed=0,
            )
            _, h_s = optimize_vqe(qnode_c, heft_va_init_fn(n, L),
                                   steps=N_ADAPTIVE_STEPS, lr=OPTIMIZER_LR)

            key2 = f"CONV_N{n}_L{L}"
            results[key2] = {
                'aheft_final' : run_a['history'][-1],
                'static_final': h_s[-1],
                'n_qubits'    : n,
                'n_layers'    : L,
            }
            print(f"  {key2}: A-H-EFT={run_a['history'][-1]:.4f} | "
                  f"Static={h_s[-1]:.4f}")

    save_results("testAT16_heisenberg_benchmark", results)
    print(f"AT16 completed in {time.time() - start_time:.2f}s")


# ===========================================================================
# ==========  PLOTTING FUNCTIONS  ===========================================
# ===========================================================================

def plot_AT1_adaptive_gv_scaling():
    """Plot AT1: Phase I vs Phase II gradient variance vs N."""
    try:
        results = load_results("testAT1_adaptive_gv_scaling")
    except FileNotFoundError:
        print("Skipping AT1 plot: results not found.")
        return

    data = []
    for key, val in results.items():
        data.append({
            'N': val['n_qubits'], 'L': val['n_layers'],
            'GV_P1': val['phase1_mean_gv'], 'GV_P2': val['phase2_mean_gv'],
        })
    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for L in LAYER_LIST:
        sub = df[df['L'] == L]
        axes[0].plot(sub['N'], sub['GV_P1'], '-o', label=f'L={L}')
        axes[1].plot(sub['N'], sub['GV_P2'], '-o', label=f'L={L}')

    for ax, title in zip(axes, ['Phase I (Static Init)', 'Phase II (sigma_crit)']):
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Number of Qubits (N)', fontsize=12)
        ax.set_ylabel(r'$\langle||\nabla C||^2\rangle$', fontsize=12)
        ax.set_title(f'AT1: GV Scaling — {title}', fontsize=13)
        ax.legend(title='L', fontsize=9, ncol=2)
        ax.grid(True, which='both', ls='--', alpha=0.4)

    plt.tight_layout()
    save_plot(fig, "AT1_Adaptive_GV_Scaling")


def plot_AT2_critical_cutoff():
    """Plot AT2: GV vs sigma with sigma_crit marked."""
    try:
        results = load_results("testAT2_critical_cutoff_sweep")
    except FileNotFoundError:
        print("Skipping AT2 plot.")
        return

    sigmas   = np.array(results['sigmas'])
    mean_gv  = np.array(results['mean_gv'])
    std_gv   = np.array(results['std_gv'])
    sig_c    = results['sigma_crit']
    n, L     = results['n_qubits'], results['n_layers']

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(sigmas, mean_gv, yerr=std_gv, fmt='-o', capsize=4,
                color='royalblue', label='A-H-EFT')
    ax.axvline(sig_c, color='red', ls='--', lw=1.5,
               label=r'$\sigma_{\rm crit}(N,L)$')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'Initialization Scale $\sigma$', fontsize=12)
    ax.set_ylabel(r'$\langle||\nabla C||^2\rangle$', fontsize=12)
    ax.set_title(f'AT2: Critical Cutoff Sweep (N={n}, L={L})', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    plt.tight_layout()
    save_plot(fig, "AT2_Critical_Cutoff_Sweep")


def plot_AT3_phase_transition():
    """Plot AT3: Gradient norm over time with Phase I/II boundary."""
    try:
        results = load_results("testAT3_phase_transition_gv")
    except FileNotFoundError:
        print("Skipping AT3 plot.")
        return

    mean_gn  = np.array(results['mean_grad_norm'])
    std_gn   = np.array(results['std_grad_norm'])
    t_switch = results['mean_t_switch']
    steps    = np.arange(len(mean_gn))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, mean_gn, color='royalblue', lw=2, label='Mean ||∇C||')
    ax.fill_between(steps, mean_gn - std_gn, mean_gn + std_gn,
                    alpha=0.2, color='royalblue')
    if t_switch > 0:
        ax.axvline(t_switch, color='darkorange', ls='--', lw=1.5,
                   label=f'Mean $t_{{switch}}$={t_switch:.0f}')
        ax.axhline(DELTA_SWITCH_DEFAULT, color='gray', ls=':', lw=1.2,
                   label=r'$\delta_{\rm switch}$')
    ax.set_yscale('log')
    ax.set_xlabel('Optimization Step', fontsize=12)
    ax.set_ylabel(r'$||\nabla C||$', fontsize=12)
    ax.set_title('AT3: Gradient Norm Across Phase I → Phase II', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    save_plot(fig, "AT3_Phase_Transition_GV")


def plot_AT4_adaptive_convergence():
    """Plot AT4: Energy trajectories for A-H-EFT vs static vs HEA."""
    try:
        results = load_results("testAT4_adaptive_convergence")
    except FileNotFoundError:
        print("Skipping AT4 plot.")
        return

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    QUBIT_SUBSET = [4, 8, 12]
    LAYER_SUBSET = [4, 8, 12]

    for i, n in enumerate(QUBIT_SUBSET):
        for j, L in enumerate(LAYER_SUBSET):
            key = f"N{n}_L{L}"
            ax  = axes[i][j]
            if key not in results:
                continue
            val = results[key]
            steps = np.arange(len(val['aheft_history']))

            ax.plot(steps, val['aheft_history'],  color='royalblue',
                    lw=2, label='A-H-EFT')
            ax.plot(steps, val['static_history'], color='green',
                    lw=1.5, ls='--', label='Static H-EFT-VA')
            ax.plot(steps, val['hea_history'],    color='darkorange',
                    lw=1.5, ls=':', label='HEA')

            ts = val.get('aheft_t_switch', -1)
            if ts > 0:
                ax.axvline(ts, color='gray', ls='-.', lw=1,
                           label=f'$t_{{switch}}$={ts}')

            ax.set_title(f'N={n}, L={L}', fontsize=11)
            ax.set_xlabel('Step', fontsize=9)
            ax.set_ylabel(r'$\langle H\rangle$', fontsize=9)
            ax.grid(True, ls='--', alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)

    plt.suptitle('AT4: Noiseless Convergence — A-H-EFT vs Baselines',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    save_plot(fig, "AT4_Adaptive_Convergence")


def plot_AT5_convergence_vs_size():
    """Plot AT5: Final energy vs N for all three methods."""
    try:
        results = load_results("testAT5_convergence_vs_system_size")
    except FileNotFoundError:
        print("Skipping AT5 plot.")
        return

    ns, e_a, e_s, e_h = [], [], [], []
    for key, val in sorted(results.items(), key=lambda x: x[1]['n_qubits']):
        ns.append(val['n_qubits'])
        e_a.append(val['aheft_final'])
        e_s.append(val['static_final'])
        e_h.append(val['hea_final'])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ns, e_a, '-o', color='royalblue',  lw=2, label='A-H-EFT')
    ax.plot(ns, e_s, '--s', color='green',     lw=2, label='Static H-EFT-VA')
    ax.plot(ns, e_h, ':^', color='darkorange', lw=2, label='HEA')
    ax.set_xlabel('Number of Qubits (N)', fontsize=12)
    ax.set_ylabel(r'Mean Final Energy $\langle H\rangle$', fontsize=12)
    ax.set_title('AT5: Convergence vs System Size (L=2)', fontsize=13)
    ax.legend(fontsize=11); ax.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    save_plot(fig, "AT5_Convergence_vs_System_Size")


def plot_AT6_fidelity():
    """Plot AT6: Ground-state fidelity with shaded error bars."""
    try:
        results = load_results("testAT6_ground_state_fidelity")
    except FileNotFoundError:
        print("Skipping AT6 plot.")
        return

    layers = results['layer_list']
    dr     = results['delta_ref']
    n      = results['n_qubits']

    fig, ax = plt.subplots(figsize=(9, 6))

    def _plot_with_shade(means, stds, color, ls, label):
        arr_m = np.array(means); arr_s = np.array(stds)
        ax.plot(layers, arr_m, ls, color=color, lw=2, label=label)
        ax.fill_between(layers, arr_m - arr_s, arr_m + arr_s,
                        alpha=0.18, color=color)

    _plot_with_shade(results['aheft_fid_mean'],  results['aheft_fid_std'],
                     'royalblue',  '-o',  'A-H-EFT')
    _plot_with_shade(results['static_fid_mean'], results['static_fid_std'],
                     'green',      '--s', 'Static H-EFT-VA')
    _plot_with_shade(results['hea_fid_mean'],    results['hea_fid_std'],
                     'darkorange', ':^',  'HEA')

    ax.set_xlabel('Circuit Depth (L)', fontsize=12)
    ax.set_ylabel(r'Ground-State Fidelity $F$', fontsize=12)
    ax.set_title(f'AT6: VQE Fidelity (N={n}, '
                 fr'$\Delta_{{\rm ref}}$={dr:.3f})', fontsize=13)
    ax.legend(fontsize=11); ax.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    save_plot(fig, "AT6_Ground_State_Fidelity")


def plot_AT7_reference_gap():
    """Plot AT7: Delta_ref vs N for TFIM and Heisenberg."""
    try:
        results = load_results("testAT7_reference_state_gap")
    except FileNotFoundError:
        print("Skipping AT7 plot.")
        return

    ns         = results['qubit_list']
    tfim_gaps  = results['tfim_gaps']
    heis_gaps  = results['heis_gaps']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, tfim_gaps,  '-o', color='royalblue',  lw=2, label='TFIM')
    ax.plot(ns, heis_gaps,  '--s', color='darkorange', lw=2, label='Heisenberg XXZ')
    ax.axhline(1.0, color='gray', ls=':', lw=1, label=r'$\Delta_{\rm ref}=1$')
    ax.set_xlabel('Number of Qubits (N)', fontsize=12)
    ax.set_ylabel(r'Reference-State Gap $\Delta_{\rm ref}$', fontsize=12)
    ax.set_title('AT7: Reference-State Gap vs System Size', fontsize=13)
    ax.legend(fontsize=11); ax.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    save_plot(fig, "AT7_Reference_State_Gap")


def plot_AT8_deff_growth():
    """Plot AT8: d_eff vs optimization step with Phase I/II boundary."""
    try:
        results = load_results("testAT8_effective_dimension_growth")
    except FileNotFoundError:
        print("Skipping AT8 plot.")
        return

    steps    = results['steps']
    deff     = results['deff']
    t_switch = results['t_switch']
    sig_c    = results['sigma_crit']
    n, L     = results['n_qubits'], results['n_layers']

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(steps, deff, '-o', color='royalblue', lw=2, label=r'$d_{\rm eff}$')
    if t_switch > 0:
        ax1.axvline(t_switch, color='darkorange', ls='--', lw=1.5,
                    label=f'$t_{{switch}}$={t_switch}')
    ax1.set_xlabel('Optimization Step', fontsize=12)
    ax1.set_ylabel(r'Effective Dimension $d_{\rm eff}$', fontsize=12,
                   color='royalblue')
    ax1.set_title(f'AT8: Hilbert Space Expansion (N={n}, L={L})', fontsize=13)
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    save_plot(fig, "AT8_Effective_Dimension_Growth")


def plot_AT9_entanglement():
    """Plot AT9: Von Neumann entropy vs L for A-H-EFT, static, HEA."""
    try:
        results = load_results("testAT9_entanglement_growth")
    except FileNotFoundError:
        print("Skipping AT9 plot.")
        return

    layers = results['layers']
    n      = results['n_qubits']

    fig, ax = plt.subplots(figsize=(8, 6))

    def _plot_shade(means, stds, color, ls, label):
        m = np.array(means); s = np.array(stds)
        ax.plot(layers, m, ls, color=color, lw=2, label=label)
        ax.fill_between(layers, m - s, m + s, alpha=0.18, color=color)

    _plot_shade(results['aheft_entropy'],  results['aheft_std'],
                'royalblue', '-o',  'A-H-EFT (Phase II)')
    _plot_shade(results['static_entropy'], results['static_std'],
                'green',     '--s', 'Static H-EFT-VA')
    _plot_shade(results['hea_entropy'],    results['hea_std'],
                'darkorange', ':^', 'HEA')

    ax.set_xlabel('Circuit Depth (L)', fontsize=12)
    ax.set_ylabel(r'Von Neumann Entropy $S_V$', fontsize=12)
    ax.set_title(f'AT9: Entanglement Growth (N={n})', fontsize=13)
    ax.legend(fontsize=11); ax.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    save_plot(fig, "AT9_Entanglement_Growth")


def plot_AT10_expressibility():
    """Plot AT10: Mean purity vs L for all methods + Haar limit."""
    try:
        results = load_results("testAT10_expressibility")
    except FileNotFoundError:
        print("Skipping AT10 plot.")
        return

    layers     = results['layer_list']
    haar_limit = results['haar_limit']
    n          = results['n_qubits']

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(layers, results['aheft_purity'],  '-o',  color='royalblue',  lw=2,
            label='A-H-EFT (Phase II)')
    ax.plot(layers, results['static_purity'], '--s', color='green',      lw=2,
            label='Static H-EFT-VA')
    ax.plot(layers, results['hea_purity'],    ':^',  color='darkorange', lw=2,
            label='HEA')
    ax.axhline(haar_limit, color='red', ls=':', lw=1.5, label='Haar Limit')

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.05, 0.95, r'Lower purity $\rightarrow$ Higher expressibility',
            transform=ax.transAxes, fontsize=10, va='top', bbox=props)

    ax.set_xlabel('Number of Layers (L)', fontsize=12)
    ax.set_ylabel(r'Mean Purity $\langle{\rm Tr}(\rho^2)\rangle$', fontsize=12)
    ax.set_title(f'AT10: Expressibility Benchmark (N={n})', fontsize=13)
    ax.legend(fontsize=11); ax.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    save_plot(fig, "AT10_Expressibility")


def plot_AT11_noise_robustness():
    """Plot AT11: Adaptive convergence under depolarizing noise."""
    try:
        results = load_results("testAT11_noise_robustness")
    except FileNotFoundError:
        print("Skipping AT11 plot.")
        return

    n, L = results['n_qubits'], results['n_layers']
    colors = ['royalblue', 'green', 'darkorange', 'red']
    labels = [k for k in results if k.startswith('P_NOISE')]

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, key in enumerate(labels):
        p     = results[key]
        hist  = p['history']
        steps = np.arange(len(hist))
        p_val = float(key.split('_')[-1])
        ts    = p['t_switch']
        ax.plot(steps, hist, color=colors[i], lw=2,
                label=f'p={p_val}')
        if ts > 0:
            ax.axvline(ts, color=colors[i], ls='-.', lw=0.8, alpha=0.6)

    ax.set_xlabel('Optimization Step', fontsize=12)
    ax.set_ylabel(r'$\langle H\rangle$', fontsize=12)
    ax.set_title(f'AT11: Noise Robustness (N={n}, L={L})', fontsize=13)
    ax.legend(title='Depolarizing p', fontsize=10)
    ax.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    save_plot(fig, "AT11_Noise_Robustness")


def plot_AT12_finite_shot():
    """Plot AT12: MSE vs shot count for A-H-EFT vs HEA."""
    try:
        results = load_results("testAT12_finite_shot_estimator")
    except FileNotFoundError:
        print("Skipping AT12 plot.")
        return

    shots_keys = [k for k in results if k.startswith('SHOTS')]
    shots_vals = [int(k.split('_')[1]) for k in shots_keys]
    aheft_mse  = [results[k]['aheft_mse'] for k in shots_keys]
    hea_mse    = [results[k]['hea_mse']   for k in shots_keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(shots_vals, aheft_mse, '-o', color='royalblue', lw=2,
              label='A-H-EFT')
    ax.loglog(shots_vals, hea_mse,   '--s', color='darkorange', lw=2,
              label='HEA')
    ax.set_xlabel('Number of Shots $M$', fontsize=12)
    ax.set_ylabel('Gradient Estimator MSE', fontsize=12)
    ax.set_title('AT12: Finite-Shot Gradient Estimator', fontsize=13)
    ax.legend(fontsize=11); ax.grid(True, which='both', ls='--', alpha=0.4)
    plt.tight_layout()
    save_plot(fig, "AT12_Finite_Shot_Estimator")


def plot_AT13_switch_sensitivity():
    """Plot AT13: Final energy vs delta_switch."""
    try:
        results = load_results("testAT13_switch_sensitivity")
    except FileNotFoundError:
        print("Skipping AT13 plot.")
        return

    keys    = sorted([k for k in results if k.startswith('DS')],
                     key=lambda x: results[x]['delta_switch'])
    ds_vals = [results[k]['delta_switch']  for k in keys]
    e_means = [results[k]['mean_e_final']  for k in keys]
    e_stds  = [results[k]['std_e_final']   for k in keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(ds_vals, e_means, yerr=e_stds, fmt='-o',
                color='royalblue', capsize=5, lw=2)
    ax.set_xscale('log')
    ax.set_xlabel(r'Switch Threshold $\delta_{\rm switch}$', fontsize=12)
    ax.set_ylabel(r'Mean Final Energy $\langle H\rangle$', fontsize=12)
    ax.set_title('AT13: Switch Criterion Sensitivity', fontsize=13)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    plt.tight_layout()
    save_plot(fig, "AT13_Switch_Sensitivity")


def plot_AT14_lambda_sensitivity():
    """Plot AT14: Final energy vs lambda."""
    try:
        results = load_results("testAT14_lambda_sensitivity")
    except FileNotFoundError:
        print("Skipping AT14 plot.")
        return

    keys     = sorted([k for k in results if k.startswith('LAM')],
                      key=lambda x: results[x]['lambda'])
    lam_vals = [results[k]['lambda']       for k in keys]
    e_means  = [results[k]['mean_e_final'] for k in keys]
    e_stds   = [results[k]['std_e_final']  for k in keys]
    clamped  = [results[k]['clamped']      for k in keys]
    sig_c    = results.get('sigma_crit', None)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(lam_vals, e_means, yerr=e_stds, fmt='-o',
                color='royalblue', capsize=5, lw=2)
    for lam, clamp in zip(lam_vals, clamped):
        if clamp:
            ax.axvline(lam, color='red', ls=':', lw=0.8, alpha=0.5)

    ax.set_xlabel(r'Growth Constant $\lambda$', fontsize=12)
    ax.set_ylabel(r'Mean Final Energy $\langle H\rangle$', fontsize=12)
    ax.set_title('AT14: Growth Constant Sensitivity', fontsize=13)
    if sig_c:
        ax.text(0.65, 0.95, f'$\\sigma_{{\\rm crit}}$={sig_c:.4f}',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    ax.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    save_plot(fig, "AT14_Lambda_Sensitivity")


def plot_AT15_statistical_significance():
    """Plot AT15: Mean energy ± std with p-value axis."""
    try:
        results = load_results("testAT15_statistical_significance")
    except FileNotFoundError:
        print("Skipping AT15 plot.")
        return

    keys = sorted(results.keys())
    ns   = [results[k]['n_qubits']    for k in keys]
    e_a  = [results[k]['aheft_mean']  for k in keys]
    s_a  = [results[k]['aheft_std']   for k in keys]
    e_s  = [results[k]['static_mean'] for k in keys]
    s_s  = [results[k]['static_std']  for k in keys]
    e_h  = [results[k]['hea_mean']    for k in keys]
    s_h  = [results[k]['hea_std']     for k in keys]
    p_vs = [results[k]['p_aheft_vs_static'] for k in keys]
    p_vh = [results[k]['p_aheft_vs_hea']    for k in keys]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(ns, e_a, yerr=s_a, fmt='-o',  color='royalblue',
                capsize=5, lw=2, label='A-H-EFT')
    ax.errorbar(ns, e_s, yerr=s_s, fmt='--s', color='green',
                capsize=5, lw=2, label='Static H-EFT-VA')
    ax.errorbar(ns, e_h, yerr=s_h, fmt=':^',  color='darkorange',
                capsize=5, lw=2, label='HEA')

    ax2 = ax.twinx()
    ax2.plot(ns, p_vs, ':D', color='purple', lw=1.5,
             label='p (A-H-EFT vs Static)')
    ax2.plot(ns, p_vh, ':x', color='red',    lw=1.5,
             label='p (A-H-EFT vs HEA)')
    ax2.axhline(0.05, color='gray', ls='-.', lw=0.8)
    ax2.set_yscale('log')
    ax2.set_ylabel('p-value', fontsize=11, color='purple')

    ax.set_xlabel('Number of Qubits (N)', fontsize=12)
    ax.set_ylabel(r'Mean Final Energy $\langle H\rangle$', fontsize=12)
    ax.set_title('AT15: Statistical Significance', fontsize=13)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              fontsize=9, loc='lower left')
    ax.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    save_plot(fig, "AT15_Statistical_Significance")


def plot_AT16_heisenberg():
    """Plot AT16: Heisenberg GV scaling Phase I vs II and convergence."""
    try:
        results = load_results("testAT16_heisenberg_benchmark")
    except FileNotFoundError:
        print("Skipping AT16 plot.")
        return

    # GV scaling
    gv_data = [(v['n_qubits'], v['n_layers'], v['phase1_gv'], v['phase2_gv'])
               for k, v in results.items() if k.startswith('N')]
    df = pd.DataFrame(gv_data, columns=['N', 'L', 'GV_P1', 'GV_P2'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for L in LAYER_LIST:
        sub = df[df['L'] == L]
        axes[0].plot(sub['N'], sub['GV_P1'], '-o', label=f'L={L}')
        axes[1].plot(sub['N'], sub['GV_P2'], '-o', label=f'L={L}')

    for ax, title in zip(axes, ['Phase I (Static)', 'Phase II (sigma_crit)']):
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('N', fontsize=12)
        ax.set_ylabel(r'$\langle||\nabla C||^2\rangle$', fontsize=12)
        ax.set_title(f'AT16 Heisenberg: {title}', fontsize=12)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, which='both', ls='--', alpha=0.4)

    plt.tight_layout()
    save_plot(fig, "AT16_Heisenberg_GV_Scaling")

    # Convergence comparison
    conv_data = {k: v for k, v in results.items() if k.startswith('CONV')}
    if conv_data:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ns_c = sorted(set(v['n_qubits'] for v in conv_data.values()))
        for n in ns_c:
            e_a_list = [v['aheft_final']  for v in conv_data.values()
                        if v['n_qubits'] == n]
            e_s_list = [v['static_final'] for v in conv_data.values()
                        if v['n_qubits'] == n]
            ls_list  = [v['n_layers']     for v in conv_data.values()
                        if v['n_qubits'] == n]
            ax2.plot(ls_list, e_a_list, '-o', lw=2,
                     label=f'A-H-EFT N={n}')
            ax2.plot(ls_list, e_s_list, '--s', lw=1.5, alpha=0.7,
                     label=f'Static N={n}')
        ax2.set_xlabel('Layers (L)', fontsize=12)
        ax2.set_ylabel(r'Final Energy $\langle H_{\rm XXZ}\rangle$', fontsize=12)
        ax2.set_title('AT16: Heisenberg Convergence', fontsize=13)
        ax2.legend(fontsize=9, ncol=2)
        ax2.grid(True, ls='--', alpha=0.4)
        plt.tight_layout()
        save_plot(fig2, "AT16_Heisenberg_Convergence")


# ===========================================================================
# --- Master Runners ---
# ===========================================================================

def run_all_tests():
    print("=" * 60)
    print("  Adaptive H-EFT-VA — Full Benchmark Suite (Paper 2)")
    print("=" * 60)
    test_AT1_adaptive_gv_scaling()
    test_AT2_critical_cutoff_sweep()
    test_AT3_phase_transition_gv()
    test_AT4_adaptive_convergence()
    test_AT5_convergence_vs_system_size()
    test_AT6_ground_state_fidelity()
    test_AT7_reference_state_gap()
    test_AT8_effective_dimension_growth()
    test_AT9_entanglement_growth()
    test_AT10_expressibility()
    test_AT11_noise_robustness()
    test_AT12_finite_shot_estimator()
    test_AT13_switch_sensitivity()
    test_AT14_lambda_sensitivity()
    test_AT15_statistical_significance()
    test_AT16_heisenberg_benchmark()
    print("\n" + "=" * 60)
    print("  All tests completed. Results saved in 'results_adaptive/'.")
    print("=" * 60)


def plot_all_results():
    print("\n" + "=" * 60)
    print("  Generating all figures for Paper 2")
    print("=" * 60)
    plot_AT1_adaptive_gv_scaling()
    plot_AT2_critical_cutoff()
    plot_AT3_phase_transition()
    plot_AT4_adaptive_convergence()
    plot_AT5_convergence_vs_size()
    plot_AT6_fidelity()
    plot_AT7_reference_gap()
    plot_AT8_deff_growth()
    plot_AT9_entanglement()
    plot_AT10_expressibility()
    plot_AT11_noise_robustness()
    plot_AT12_finite_shot()
    plot_AT13_switch_sensitivity()
    plot_AT14_lambda_sensitivity()
    plot_AT15_statistical_significance()
    plot_AT16_heisenberg()
    print("\nAll figures saved in 'figures_adaptive/'.")
    print("=" * 60)


if __name__ == '__main__':
    # Requirements: pennylane numpy matplotlib seaborn pandas scipy
    # Install:      pip install pennylane numpy matplotlib seaborn pandas scipy
    run_all_tests()
    plot_all_results()