"""
Experiment: Genetic Algorithm for the Travelling Salesman Problem (10 cities)
 
Steps per generation:
  1. Evaluate  : tour length and fitness (f = 1 / length)
  2. Select    : roulette wheel (probability proportional to fitness)
  3. Crossover : two-point order crossover (rate 0.7)
  4. Mutation  : swap two cities (rate 0.2)
  5. Elitism   : best tour always survives to the next generation
"""
import random
 
# ---------------- Parameters ----------------
POPULATION_SIZE = 20
NUM_GENERATIONS = 50  # default; overwritten by user input in main
P_CROSSOVER = 0.7
P_MUTATION = 0.2
 
# ---------------- Problem data ----------------
DIST = [
    [0, 14, 12, 18, 25, 31, 9, 15, 22, 19],
    [14, 0, 11, 20, 28, 17, 13, 24, 16, 21],
    [12, 11, 0, 15, 19, 23, 27, 10, 14, 26],
    [18, 20, 15, 0, 13, 16, 22, 29, 12, 17],
    [25, 28, 19, 13, 0, 14, 21, 18, 11, 15],
    [31, 17, 23, 16, 14, 0, 19, 25, 20, 12],
    [9, 13, 27, 22, 21, 19, 0, 16, 24, 18],
    [15, 24, 10, 29, 18, 25, 16, 0, 13, 22],
    [22, 16, 14, 12, 11, 20, 24, 13, 0, 14],
    [19, 21, 26, 17, 15, 12, 18, 22, 14, 0],
]
N = len(DIST)
 
 
# ---------------- Helper functions ----------------
def show(route):
    """Print route with cities numbered 1..N instead of 0..N-1."""
    return str([c + 1 for c in route])
 
 
def tour_cost(route):
    """Length of the closed tour (returns to the starting city)."""
    return sum(DIST[route[k]][route[(k + 1) % N]] for k in range(N))
 
 
def fitness_of(route):
    """Shorter tour -> higher fitness."""
    return 1.0 / tour_cost(route)
 
 
def spin_wheel(pop, fits):
    """Roulette wheel selection."""
    spin = random.uniform(0, sum(fits))
    running = 0.0
    for route, f in zip(pop, fits):
        running += f
        if running >= spin:
            return route
    return pop[-1]
 
 
def build_child(keep_part, lo, hi, other_parent):
    """
    Keep keep_part[lo:hi] in place, then fill the empty slots
    with the missing cities in the order they appear in other_parent.
    """
    child = [None] * N
    child[lo:hi] = keep_part[lo:hi]
    leftovers = iter(c for c in other_parent if c not in child)
    for pos in range(N):
        if child[pos] is None:
            child[pos] = next(leftovers)
    return child
 
 
def recombine(parent_a, parent_b):
    """Two-point order crossover, applied with probability P_CROSSOVER."""
    if random.random() >= P_CROSSOVER:
        return parent_a[:], parent_b[:]
    lo, hi = sorted(random.sample(range(N), 2))
    return (build_child(parent_a, lo, hi, parent_b),
            build_child(parent_b, lo, hi, parent_a))
 
 
def swap_mutation(route):
    """With probability P_MUTATION, swap two random cities."""
    route = route[:]
    if random.random() < P_MUTATION:
        a, b = random.sample(range(N), 2)
        route[a], route[b] = route[b], route[a]
    return route
 
 
# ---------------- Main GA loop ----------------
def run(NUM_GENERATIONS=NUM_GENERATIONS):
    population = [random.sample(range(N), N) for _ in range(POPULATION_SIZE)]
 
    for gen in range(1, NUM_GENERATIONS + 1):
        fits = [fitness_of(r) for r in population]
        elite = min(population, key=tour_cost)
 
        print("\n" + "=" * 80)
        print(f"GENERATION {gen}")
        print("=" * 80)
 
        # ---- Table 1 ----
        print("\nTABLE 1: POPULATION AND FITNESS")
        print(f"{'No.':<5}{'Route':<35}{'Distance':<12}{'Fitness':<10}")
        print("-" * 62)
        for idx, (r, f) in enumerate(zip(population, fits), start=1):
            print(f"{idx:<5}{show(r):<35}{tour_cost(r):<12}{f:<10.4f}")
 
        # ---- Table 2 ----
        print("\nTABLE 2: SELECTION, CROSSOVER AND MUTATION")
        print(f"{'No.':<5}{'Parent 1':<36}{'Parent 2':<36}"
              f"{'Child 1':<36}{'Child 2':<36}")
        print("-" * 150)
 
        next_gen = [elite]
        step = 1
        while len(next_gen) < POPULATION_SIZE:
            p1 = spin_wheel(population, fits)
            p2 = spin_wheel(population, fits)
            kid1, kid2 = recombine(p1, p2)
            kid1, kid2 = swap_mutation(kid1), swap_mutation(kid2)
 
            print(f"{step:<5}{show(p1):<36}{show(p2):<36}"
                  f"{show(kid1):<36}{show(kid2):<36}")
 
            next_gen.append(kid1)
            if len(next_gen) < POPULATION_SIZE:
                next_gen.append(kid2)
            step += 1
 
        population = next_gen
 
        # ---- Table 3 ----
        print("\nTABLE 3: NEXT GENERATION")
        print(f"{'No.':<5}{'Route':<35}{'Distance':<12}{'Status':<10}")
        print("-" * 62)
        for idx, r in enumerate(population, start=1):
            tag = "Elite" if idx == 1 else "Selected"
            print(f"{idx:<5}{show(r):<35}{tour_cost(r):<12}{tag:<10}")
 
        best = min(population, key=tour_cost)
        print(f"\nBEST ROUTE OF GENERATION {gen}")
        print("Best Route:", [c + 1 for c in best] + [best[0] + 1])
        print("Minimum Distance:", tour_cost(best))
 
    best = min(population, key=tour_cost)
    print("\n" + "=" * 50)
    print("FINAL OPTIMAL SOLUTION")
    print("=" * 50)
    print("Best Route:", [c + 1 for c in best] + [best[0] + 1])
    print("Minimum Distance:", tour_cost(best))
 
 
if __name__ == "__main__":
    while True:
        try:
            gens = int(input("Enter number of generations: "))
            if gens > 0:
                break
            print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter an integer.")
    run(gens)