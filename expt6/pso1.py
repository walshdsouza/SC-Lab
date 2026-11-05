import numpy as np


def objective_function(x):
    return x * np.sin(10 * np.pi * x) + 2


def find_local_extrema(positions, fitness):
    sorted_indices = np.argsort(positions)
    results = ["Neither"] * len(positions)

    for i in range(1, len(positions) - 1):
        left = sorted_indices[i - 1]
        current = sorted_indices[i]
        right = sorted_indices[i + 1]

        if (fitness[current] > fitness[left] and
                fitness[current] > fitness[right]):
            results[current] = "Local Maximum"
        elif (fitness[current] < fitness[left] and
              fitness[current] < fitness[right]):
            results[current] = "Local Minimum"

    return results


def particle_swarm_optimization(
    n_particles=10,
    max_iterations=10,
    lower_bound=-1,
    upper_bound=2,
    w=0.7,
    c1=1.5,
    c2=1.5,
    seed=42
):
    rng = np.random.default_rng(seed)

    positions = rng.uniform(lower_bound, upper_bound, n_particles)
    velocities = rng.uniform(-0.1, 0.1, n_particles)
    fitness = objective_function(positions)

    personal_best_positions = positions.copy()
    personal_best_fitness = fitness.copy()

    best_index = np.argmax(fitness)
    global_best_position = positions[best_index]
    global_best_fitness = fitness[best_index]

    print("Initial Population")
    print("Particle\tPosition (x)\tFitness")
    for i in range(n_particles):
        print(
            f"{i+1}\t\t{positions[i]:.6f}\t"
            f"{fitness[i]:.6f}"
        )
    print(f"Initial global best = {global_best_fitness:.6f}")

    max_velocity = 0.2 * (upper_bound - lower_bound)

    for iteration in range(max_iterations):
        r1 = rng.random(n_particles)
        r2 = rng.random(n_particles)

        velocities = (
            w * velocities
            + c1 * r1 * (personal_best_positions - positions)
            + c2 * r2 * (global_best_position - positions)
        )
        velocities = np.clip(velocities, -max_velocity, max_velocity)

        positions = np.clip(
            positions + velocities,
            lower_bound,
            upper_bound
        )
        fitness = objective_function(positions)

        improved = fitness > personal_best_fitness
        personal_best_positions[improved] = positions[improved]
        personal_best_fitness[improved] = fitness[improved]

        best_index = np.argmax(personal_best_fitness)
        if personal_best_fitness[best_index] > global_best_fitness:
            global_best_fitness = personal_best_fitness[best_index]
            global_best_position = personal_best_positions[best_index]

        print(f"\nIteration {iteration + 1}")
        print("Particle\tPosition (x)\tFitness")
        for i in range(n_particles):
            print(
                f"{i+1}\t\t{positions[i]:.6f}\t"
                f"{fitness[i]:.6f}"
            )

        results = find_local_extrema(positions, fitness)
        print("\nParticle-wise Local Maxima and Minima")
        for i in range(n_particles):
            print(
                f"Particle {i+1}: "
                f"x = {positions[i]:.6f}, "
                f"Fitness = {fitness[i]:.6f}, "
                f"Result = {results[i]}"
            )

        print(f"\nGlobal best x = {global_best_position:.8f}")
        print(f"Maximum fitness = {global_best_fitness:.8f}")

    print("\nFinal Results")
    print(f"Optimal x = {global_best_position:.10f}")
    print(f"Maximum fitness = {global_best_fitness:.10f}")

    return global_best_position, global_best_fitness


if __name__ == "__main__":
    best_x, best_fitness = particle_swarm_optimization(
        n_particles=10,
        max_iterations=10,
        lower_bound=-1,
        upper_bound=2,
        w=0.7,
        c1=1.5,
        c2=1.5,
        seed=42
    )