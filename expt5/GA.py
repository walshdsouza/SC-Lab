import random
import math

POP_SIZE = 20
BITS = 10
GENERATIONS = 5
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1

def decode(chromosome):
    return int(chromosome, 2)

def fitness(chromosome):
    x = int(chromosome, 2) / 1023
    return x * math.sin(10 * math.pi * x) + 2

def roulette_selection(population, fitness_values):
    total = sum(fitness_values)
    pick = random.uniform(0, total)
    current = 0
    for chromosome, value in zip(population, fitness_values):
        current += value
        if current >= pick:
            return chromosome
    return population[-1]

def crossover(p1, p2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, BITS - 1)
        c1 = p1[:point] + p2[point:]
        c2 = p2[:point] + p1[point:]
        return c1, c2, point
    return p1, p2, None

def mutate(chromosome):
    bits = list(chromosome)
    changed = []
    for i in range(BITS):
        if random.random() < MUTATION_RATE:
            bits[i] = '1' if bits[i] == '0' else '0'
            changed.append(i + 1)
    return "".join(bits), changed

population = [format(random.randint(0, 1023), '010b') for _ in range(POP_SIZE)]

for generation in range(1, GENERATIONS + 1):
    fitness_values = [fitness(c) for c in population]
    total_fitness = sum(fitness_values)
    probabilities = [f / total_fitness for f in fitness_values]
    elite = max(population, key=fitness)
    
    print(f"\nGENERATION {generation}")
    print("\nTABLE 1: POPULATION AND FITNESS")
    print(f"{'No.':<5} {'Chromosome':<15} {'x':<8} {'Fitness':<15} {'Probability':<15} {'Expected Count':<15}")
    print("-" * 73)
    
    for i, chromosome in enumerate(population):
        print(f"{i+1:<5} {chromosome:<15} {decode(chromosome):<8} {fitness_values[i]:<15.4f} {probabilities[i]:<15.4f} {POP_SIZE * probabilities[i]:<15.4f}")
        
    print("\nTABLE 2: SELECTION, CROSSOVER AND MUTATION")
    print(f"{'No.':<5} {'Parent 1':<13} {'Parent 2':<13} {'Point':<8} {'Child 1':<13} {'Child 2':<13} {'Mutated Child 1':<18} {'Mutated Child 2':<18}")
    print("-" * 101)
    
    new_population = [elite]
    operation_no = 1
    
    while len(new_population) < POP_SIZE:
        p1 = roulette_selection(population, fitness_values)
        p2 = roulette_selection(population, fitness_values)
        c1, c2, point = crossover(p1, p2)
        m1, changes1 = mutate(c1)
        m2, changes2 = mutate(c2)
        
        print(f"{operation_no:<5} {p1:<13} {p2:<13} {str(point) if point else '-':<8} {c1:<13} {c2:<13} {m1:<18} {m2:<18}")
        
        new_population.append(m1)
        if len(new_population) < POP_SIZE:
            new_population.append(m2)
        operation_no += 1
        
    population = new_population[:POP_SIZE]
    
    print("\nTABLE 3: POPULATION FOR NEXT GENERATION")
    print(f"{'No.':<5} {'Chromosome':<15} {'x':<8} {'Fitness':<15} {'Status':<15}")
    print("-" * 58)
    
    for i, chromosome in enumerate(population):
        status = "Elite" if chromosome == elite and i == 0 else "Selected"
        print(f"{i+1:<5} {chromosome:<15} {decode(chromosome):<8} {fitness(chromosome):<15.4f} {status:<15}")

best = max(population, key=fitness)
print("\nFINAL OPTIMAL SOLUTION")
print("Chromosome:", best)
print("x =", decode(best))
print("Maximum Fitness =", round(fitness(best), 4))