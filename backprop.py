import math

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + math.exp(-x))

def build_and_train_network():
    print("=== DYNAMIC BACKPROPAGATION NETWORK SETUP ===\n")
    
    # 1. Network Architecture
    try:
        n_in = int(input("Enter number of Input nodes: "))
        n_hid = int(input("Enter number of Hidden nodes: "))
        n_out = int(input("Enter number of Output nodes: "))
        
        # 2. Input and Target Values
        print("\n--- Initial Values ---")
        X = [float(input(f"Value for Input node {i+1}: ")) for i in range(n_in)]
        T = [float(input(f"Target value for Output node {k+1}: ")) for k in range(n_out)]
        
        # 3. Connection Weights
        print("\n--- Weights: Input Layer to Hidden Layer ---")
        # V[i][j] represents weight from input i to hidden j
        V = []
        for i in range(n_in):
            row = []
            for j in range(n_hid):
                weight = float(input(f"Weight from Input {i+1} to Hidden {j+1}: "))
                row.append(weight)
            V.append(row)
            
        print("\n--- Weights: Hidden Layer to Output Layer ---")
        # W[j][k] represents weight from hidden j to output k
        W = []
        for j in range(n_hid):
            row = []
            for k in range(n_out):
                weight = float(input(f"Weight from Hidden {j+1} to Output {k+1}: "))
                row.append(weight)
            W.append(row)
            
        # 4. Training Parameters
        print("\n--- Training Parameters ---")
        eta = float(input("Enter learning rate (e.g., 1.0): "))
        error_threshold = float(input("Enter error threshold (e.g., 0.001): "))
        max_epochs = int(input("Enter maximum epochs (e.g., 1000): "))
        
    except ValueError:
        print("\n[!] Invalid input. Please run the script again and enter numerical values.")
        return

    print("\n" + "="*40)
    print("STARTING TRAINING...")
    print("="*40)

    epoch = 1
    current_error = float('inf')

    # Main Training Loop
    while current_error > error_threshold and epoch <= max_epochs:
        print(f"\n{'-'*15} EPOCH {epoch} {'-'*15}")
        
        # --- FORWARD PASS ---
        # Calculate Hidden Layer outputs
        H = []
        for j in range(n_hid):
            net_j = sum(X[i] * V[i][j] for i in range(n_in))
            H.append(sigmoid(net_j))
            
        # Calculate Final Outputs
        O = []
        for k in range(n_out):
            net_k = sum(H[j] * W[j][k] for j in range(n_hid))
            O.append(sigmoid(net_k))
            
        # Calculate Total Error (Mean Squared Error)
        current_error = 0.5 * sum((T[k] - O[k])**2 for k in range(n_out))
        
        print("-> Forward Pass Results:")
        for j in range(n_hid):
            print(f"   Hidden Node {j+1} Output: {H[j]:.5f}")
        for k in range(n_out):
            print(f"   Final Output Node {k+1}: {O[k]:.5f}")
        print(f"   Total Network Error:    {current_error:.5f}")
        
        if current_error <= error_threshold:
            print(f"\n[SUCCESS] Target error threshold ({error_threshold}) reached in {epoch} epochs!")
            break
            
        # --- BACKWARD PASS ---
        print("\n-> Backward Pass (Calculating Gradients):")
        
        # Output Layer Deltas
        delta_O = []
        for k in range(n_out):
            delta = (T[k] - O[k]) * O[k] * (1 - O[k])
            delta_O.append(delta)
            print(f"   Output Node {k+1} Delta: {delta:.6f}")
            
        # Hidden Layer Deltas
        delta_H = []
        for j in range(n_hid):
            error_sum = sum(delta_O[k] * W[j][k] for k in range(n_out))
            delta = error_sum * H[j] * (1 - H[j])
            delta_H.append(delta)
            print(f"   Hidden Node {j+1} Delta: {delta:.6f}")
            
        # --- WEIGHT UPDATES ---
        print("\n-> Weight Updates:")
        
        # Update Hidden-to-Output Weights (W)
        for j in range(n_hid):
            for k in range(n_out):
                W[j][k] = W[j][k] + (eta * delta_O[k] * H[j])
                print(f"   New W(Hidden {j+1} -> Output {k+1}): {W[j][k]:.5f}")
                
        # Update Input-to-Hidden Weights (V)
        for i in range(n_in):
            for j in range(n_hid):
                V[i][j] = V[i][j] + (eta * delta_H[j] * X[i])
                print(f"   New V(Input {i+1} -> Hidden {j+1}): {V[i][j]:.5f}")
        
        epoch += 1

    if epoch > max_epochs:
        print(f"\n[STOPPED] Maximum epochs ({max_epochs}) reached. The network did not converge to the error threshold.")
        print(f"Final Error: {current_error:.5f}")

if __name__ == "__main__":
    build_and_train_network()