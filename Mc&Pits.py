# McCulloch-Pitts Neural Network
# 3 Inputs: X1, X2, X3
# Gates: AND, OR, NAND, NOR, XOR


def mp_neuron(x1, x2, x3, w1, w2, w3, threshold):
    total = (x1 * w1) + (x2 * w2) + (x3 * w3)

    if total >= threshold:
        return 1
    else:
        return 0


# ---------------- XOR FUNCTION ----------------
# 3-input XOR gives 1 when the number of 1s is ODD.
#
# 000 -> 0
# 001 -> 1
# 010 -> 1
# 011 -> 0
# 100 -> 1
# 101 -> 0
# 110 -> 0
# 111 -> 1

def xor_3_input(x1, x2, x3):

    # Hidden neurons detect exactly one 1

    h1 = mp_neuron(x1, x2, x3, 1, -1, -1, 1)
    h2 = mp_neuron(x1, x2, x3, -1, 1, -1, 1)
    h3 = mp_neuron(x1, x2, x3, -1, -1, 1, 1)

    # Hidden neuron detects all three 1s
    h4 = mp_neuron(x1, x2, x3, 1, 1, 1, 3)

    # Output neuron performs OR on hidden outputs
    total = h1 + h2 + h3 + h4

    if total >= 1:
        return 1
    else:
        return 0


# ---------------- MAIN PROGRAM ----------------

print("McCulloch-Pitts Neural Network")
print("1. AND")
print("2. OR")
print("3. NAND")
print("4. NOR")
print("5. XOR")

choice = int(input("Enter gate choice: "))

x1 = int(input("Enter X1 (0 or 1): "))
x2 = int(input("Enter X2 (0 or 1): "))
x3 = int(input("Enter X3 (0 or 1): "))


# Validate input
if x1 not in [0, 1] or x2 not in [0, 1] or x3 not in [0, 1]:
    print("Inputs must be 0 or 1.")
    exit()


# AND
if choice == 1:

    w1 = 1
    w2 = 1
    w3 = 1
    threshold = 3

    output = mp_neuron(x1, x2, x3, w1, w2, w3, threshold)

    print("\nGate: AND")
    print("Weights:", w1, w2, w3)
    print("Threshold:", threshold)
    print("Output:", output)


# OR
elif choice == 2:

    w1 = 1
    w2 = 1
    w3 = 1
    threshold = 1

    output = mp_neuron(x1, x2, x3, w1, w2, w3, threshold)

    print("\nGate: OR")
    print("Weights:", w1, w2, w3)
    print("Threshold:", threshold)
    print("Output:", output)


# NAND
elif choice == 3:

    w1 = -1
    w2 = -1
    w3 = -1
    threshold = -2

    output = mp_neuron(x1, x2, x3, w1, w2, w3, threshold)

    print("\nGate: NAND")
    print("Weights:", w1, w2, w3)
    print("Threshold:", threshold)
    print("Output:", output)


# NOR
elif choice == 4:

    w1 = -1
    w2 = -1
    w3 = -1
    threshold = 0

    output = mp_neuron(x1, x2, x3, w1, w2, w3, threshold)

    print("\nGate: NOR")
    print("Weights:", w1, w2, w3)
    print("Threshold:", threshold)
    print("Output:", output)


# XOR
elif choice == 5:

    output = xor_3_input(x1, x2, x3)

    print("\nGate: 3-Input XOR")
    print("Output:", output)


else:
    print("Invalid choice!")