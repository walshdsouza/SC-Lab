def mp_neuron(x1, x2, x3, w1, w2, w3, threshold):
    total = (x1 * w1) + (x2 * w2) + (x3 * w3)

    if total >= threshold:
        return 1
    else:
        return 0

def mp_neuron_1(x, w, threshold):
    total = x * w
    
    if total >= threshold:
        return 1
    else:
        return 0

def nor_3_input(x1, x2, x3, w1, w2, w3):
    h1 = mp_neuron(x1, x2, x3, w1, w2, w3, 1)
    
    output = mp_neuron_1(h1, -1, 0)
    
    return output

def nand_3_input(x1, x2, x3, w1, w2, w3):
    h1 = mp_neuron_1(x1, w1, 0)
    h2 = mp_neuron_1(x2, w2, 0)
    h3 = mp_neuron_1(x3, w3, 0)
    
    output = mp_neuron(h1, h2, h3, 1, 1, 1, 1)
    
    return output

def xor_3_input(x1, x2, x3, w1, w2, w3):
    n1 = mp_neuron(x1, x2, x3, w1, w2, w3, 1)
    n2 = mp_neuron(x1, x2, x3, w1, w2, w3, 2)
    n3 = mp_neuron(x1, x2, x3, w1, w2, w3, 3)
    
    intermediate = mp_neuron(n1, n2, 0, 1, -2, 0, 1)
    
    output = mp_neuron(intermediate, n3, 0, 1, 1, 0, 1)
    
    return output

print("McCulloch-Pitts Neural Network")
print("1. AND")
print("2. OR")
print("3. NAND")
print("4. NOR")
print("5. XOR")

choice = int(input("Enter gate choice (1-5): "))

x1 = int(input("Enter X1 (0 or 1): "))
x2 = int(input("Enter X2 (0 or 1): "))
x3 = int(input("Enter X3 (0 or 1): "))

if x1 not in [0, 1] or x2 not in [0, 1] or x3 not in [0, 1]:
    print("Inputs must be 0 or 1.")
    exit()

if choice in [1, 2, 3, 4, 5]:
    print("\n--- Enter Weights ---")
    w1 = float(input("Enter weight W1: "))
    w2 = float(input("Enter weight W2: "))
    w3 = float(input("Enter weight W3: "))
    
    if choice == 1:
        gate_name = "AND"
        threshold = 3
        output = mp_neuron(x1, x2, x3, w1, w2, w3, threshold)
        print(f"\nGate: {gate_name}")
        print(f"Weights: {w1}, {w2}, {w3}")
        print(f"Threshold: {threshold}")
        print(f"Output: {output}")
        
    elif choice == 2:
        gate_name = "OR"
        threshold = 1
        output = mp_neuron(x1, x2, x3, w1, w2, w3, threshold)
        print(f"\nGate: {gate_name}")
        print(f"Weights: {w1}, {w2}, {w3}")
        print(f"Threshold: {threshold}")
        print(f"Output: {output}")

    elif choice == 3:
        output = nand_3_input(x1, x2, x3, w1, w2, w3)
        print("\nGate: 3-Input NAND")
        print(f"Weights: {w1}, {w2}, {w3}")
        print(f"Output: {output}")

    elif choice == 4:
        output = nor_3_input(x1, x2, x3, w1, w2, w3)
        print("\nGate: 3-Input NOR")
        print(f"Weights: {w1}, {w2}, {w3}")
        print(f"Output: {output}")

    elif choice == 5:
        output = xor_3_input(x1, x2, x3, w1, w2, w3)
        print("\nGate: 3-Input XOR")
        print(f"Weights: {w1}, {w2}, {w3}")
        print(f"Output: {output}")

else:
    print("Invalid choice!")