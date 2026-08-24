def mp_neuron(x1, x2, x3, w1, w2, w3, threshold):
    total = (x1 * w1) + (x2 * w2) + (x3 * w3)

    if total >= threshold:
        return 1
    else:
        return 0


def xor_3_input(x1, x2, x3):
    
    h1 = mp_neuron(x1, x2, x3, 1, -1, -1, 1)
    h2 = mp_neuron(x1, x2, x3, -1, 1, -1, 1)
    h3 = mp_neuron(x1, x2, x3, -1, -1, 1, 1)

    
    h4 = mp_neuron(x1, x2, x3, 1, 1, 1, 3)

    
    total = h1 + h2 + h3 + h4

    if total >= 1:
        return 1
    else:
        return 0



print("McCulloch-Pitts Neural Network")
print("1. AND")
print("2. OR")
print("3. NAND")
print("4. NOR")
print("5. XOR (Predefined Network)")

choice = int(input("Enter gate choice (1-5): "))

x1 = int(input("Enter X1 (0 or 1): "))
x2 = int(input("Enter X2 (0 or 1): "))
x3 = int(input("Enter X3 (0 or 1): "))


if x1 not in [0, 1] or x2 not in [0, 1] or x3 not in [0, 1]:
    print("Inputs must be 0 or 1.")
    exit()


if choice in [1, 2, 3, 4]:
    print("\n--- Enter Weights ---")
    w1 = float(input("Enter weight W1: "))
    w2 = float(input("Enter weight W2: "))
    w3 = float(input("Enter weight W3: "))
    
    
    if choice == 1:
        gate_name = "AND"
        threshold = 3
    elif choice == 2:
        gate_name = "OR"
        threshold = 1
    elif choice == 3:
        gate_name = "NAND"
        threshold = 1
    elif choice == 4:
        gate_name = "NOR"
        threshold = 0

    output = mp_neuron(x1, x2, x3, w1, w2, w3, threshold)
    
    print(f"\nGate: {gate_name}")
    print(f"Weights: {w1}, {w2}, {w3}")
    print(f"Threshold: {threshold} ")
    print(f"Output: {output}")


elif choice == 5:
    output = xor_3_input(x1, x2, x3)

    print("\nGate: 3-Input XOR")
    print(f"Output: {output}")

else:
    print("Invalid choice!")