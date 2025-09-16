import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# =========================
# Benchmark: SCA All Code
# =========================
def f1_sphere(x: np.ndarray) -> float:
    """
    f1(x) = sum_{i=1..D} x_i^2
    """
    x = np.asarray(x, dtype=float)
    return float(np.sum(x**2))

def f2_rosenbrock(x: np.ndarray) -> float:
    """
    f2(x1, x2) = 100*(x2 - x1**2)**2 + (1 - x1)**2
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    x1, x2 = x
    return float(100.0 * (x1**2-x2)**2 + (1.0 - x1)**2)

def f3_rastrigin(x: np.ndarray) -> float:
    """
    Rastrigin function (f3): sum[x_i^2 - 10 cos(2πx_i) + 10].
    """
    x = np.asarray(x, dtype=float)
    return float(np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x) + 10.0))

def f4_griewank(x: np.ndarray) -> float:
    """
    f4(x) = 1 + sum_{i=1..D} (x_i^2 / 4000) - prod_{i=1..D} cos(x_i / sqrt(i))
    """
    x = np.asarray(x, dtype=float)
    s = np.sum(x**2) / 4000.0
    i = np.arange(1, x.size + 1, dtype=float)
    p = np.prod(np.cos(x / np.sqrt(i)))
    return 1.0 + s - p


# =========================
# Sine Cosine Algorithm
# =========================
def sca(objective_function, lb, ub, dim, num_agents, max_iter, seed=None):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(lb, ub, size=(num_agents, dim))
    dest_pos = np.zeros(dim)
    dest_fitness = float("inf")

    # --- Logging variables ---
    convergence = np.zeros(max_iter)
    pop_history_2d = []
    avg_fitness_history = np.zeros(max_iter)
    first_agent_traj_x1 = np.zeros(max_iter)
    full_pop_history = []

    a = 2.0
    for t in range(max_iter):
        # Evaluate
        positions = np.clip(positions, lb, ub)
        fitnesses = np.apply_along_axis(objective_function, 1, positions)

        # --- Log data for this iteration ---
        avg_fitness_history[t] = np.mean(fitnesses)
        first_agent_traj_x1[t] = positions[0, 0]
        full_pop_history.append(positions.copy())

        # Update best
        idx = np.argmin(fitnesses)
        if fitnesses[idx] < dest_fitness:
            dest_fitness = float(fitnesses[idx])
            dest_pos = positions[idx].copy()

        convergence[t] = dest_fitness

        # Save for 2D slides (first 20 iterations)
        if dim >= 2 and t < 20:
            pop_history_2d.append(positions[:, :2].copy())

        # Move agents
        r1 = a - t * (a / max_iter)
        for i in range(num_agents):
            for j in range(dim):
                r2 = rng.uniform(0, 2 * np.pi)
                r3 = rng.uniform(0, 2)
                r4 = rng.uniform(0, 1)
                if r4 < 0.5:
                    positions[i, j] = positions[i, j] + r1 * np.sin(r2) * abs(r3 * dest_pos[j] - positions[i, j])
                else:
                    positions[i, j] = positions[i, j] + r1 * np.cos(r2) * abs(r3 * dest_pos[j] - positions[i, j])

    logs = {
        "convergence": convergence,
        "pop_history_2d": pop_history_2d,
        "avg_fitness": avg_fitness_history,
        "first_agent_traj_x1": first_agent_traj_x1,
        "full_pop_history": full_pop_history
    }
    return dest_pos, dest_fitness, logs

def run_multiple_trials(objective_function, lb, ub, dim, num_agents, max_iter, num_trials=10):
    all_curves = []
    final_mins = []

    for trial in range(num_trials):
        _, best_fitness, logs = sca(objective_function, lb, ub, dim, num_agents, max_iter, seed=1234+trial)
        all_curves.append(logs["convergence"])
        final_mins.append(best_fitness)
        print(f"[Trial {trial+1:02d}] best min = {best_fitness:.6e}")

    all_curves = np.array(all_curves)
    mean_curve = np.mean(all_curves, axis=0)
    std_curve = np.std(all_curves, axis=0)

    print("\n=== Result ===")
    print(f"Mean final min = {np.mean(final_mins):.6e}")
    print(f"Std  final min = {np.std(final_mins, ddof=1):.6e}")
    print("\n")

    return mean_curve, std_curve

def f1_plots_for_D2():
    D = 2
    lb, ub = -5.0, 5.0
    num_agents = 20
    max_iter = 120
    seed = 2025
    _, best, _ = sca(f1_sphere, lb, ub, D, num_agents, max_iter, seed=seed)
    print(f"[D=2] Sphere best min = {best:.6e}")

def f2_plots_for_D2():
    D = 2
    lb, ub = -2.0, 2.0
    num_agents = 20
    max_iter = 120
    seed = 2025
    _, best, _ = sca(f2_rosenbrock, lb, ub, D, num_agents, max_iter, seed=seed)
    print(f"[D=2] Rosenbrock best min = {best:.6e}")

def f3_plots_for_D2():
    D = 2
    lb, ub = -5.0, 5.0
    num_agents = 20
    max_iter = 120
    seed = 2025
    _, best, _ = sca(f3_rastrigin, lb, ub, D, num_agents, max_iter, seed=seed)
    print(f"[D=2] Rastrigin best min = {best:.6e}")

def f4_plots_for_D2():
    D = 2
    lb, ub = -600.0, 600.0
    num_agents = 20
    max_iter = 120
    seed = 2025
    _, best, _ = sca(f4_griewank, lb, ub, D, num_agents, max_iter, seed=seed)
    print(f"[D=2] Griewank best min = {best:.6e}")


if __name__ == "__main__":

    print("Running Benchmark: F1 - Sphere Function")
    objective = f1_sphere
    D = 30
    lb, ub = -5.0, 5.0
    num_agents = 20
    max_iter = 300
    num_trials = 10
    mean_curve, std_curve = run_multiple_trials(
        objective, lb, ub, D, num_agents, max_iter, num_trials
    )
    f1_plots_for_D2()
    print("\n")
    
    # ========== F2: Rosenbrock ==========
    print("Running Benchmark: F2 - Rosenbrock Function")
    objective = f2_rosenbrock
    D = 2
    lb, ub = -2.0, 2.0
    num_agents = 20
    max_iter = 300
    num_trials = 10
    mean_curve, std_curve = run_multiple_trials(
        objective, lb, ub, D, num_agents, max_iter, num_trials
    )
    f2_plots_for_D2()
    print("\n")

    # ========== F3: Rastrigin ==========
    print("Running Benchmark: F3 - Rastrigin Function")
    objective = f3_rastrigin
    D = 20
    lb, ub = -5.0, 5.0
    num_agents = 20
    max_iter = 300
    num_trials = 10
    mean_curve, std_curve = run_multiple_trials(
        objective, lb, ub, D, num_agents, max_iter, num_trials
    )
    f3_plots_for_D2()
    print("\n")

    # ========== F4: Griewank ==========
    print("Running Benchmark: F4 - Griewank Function")
    objective = f4_griewank
    D = 30
    lb, ub = -600.0, 600.0
    num_agents = 20
    max_iter = 300
    num_trials = 10
    mean_curve, std_curve = run_multiple_trials(
        objective, lb, ub, D, num_agents, max_iter, num_trials
    )
    f4_plots_for_D2()

    print("\nSuccessful")
