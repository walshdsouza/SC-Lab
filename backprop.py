import math

def sigmoid(x):
    """Sigmoid activation function with lambda=1."""
    return 1 / (1 + math.exp(-x))

def main():
    print("=== BACKPROPAGATION NETWORK (ERROR THRESHOLD) ===")
    
    # 1. Network Architecture Setup
    try:
        n_in = int(input("Enter number of Input nodes: "))
        n_hid = int(input("Enter number of Hidden nodes: "))
        n_out = int(input("Enter number of Output nodes: "))
        
        x = []
        for i in range(n_in):
            x.append(float(input(f"Enter Input x{i+1}: ")))
            
        d = []
        for k in range(n_out):
            d.append(float(input(f"Enter Target output d{k+1}: ")))
            
        eta = float(input("Enter Learning Rate (eta): "))
        error_threshold = float(input("Enter target error threshold (e.g., 0.005): "))
        max_epochs = int(input("Enter max epochs (to prevent infinite loops, e.g., 1000): "))
        
        # 2. Weight Initialization
        print("\n--- Initial Weights (Input to Hidden) ---")
        v = [[0 for _ in range(n_hid)] for _ in range(n_in)]
        for i in range(n_in):
            for j in range(n_hid):
                v[i][j] = float(input(f"Enter weight v_{i+1}{j+1} (Input {i+1} -> Hidden {j+1}): "))
                
        print("\n--- Initial Weights (Hidden to Output) ---")
        w = [[0 for _ in range(n_out)] for _ in range(n_hid)]
        for j in range(n_hid):
            for k in range(n_out):
                w[j][k] = float(input(f"Enter weight w_{j+1}{k+1} (Hidden {j+1} -> Output {k+1}): "))

    except ValueError:
        print("\n[!] Invalid input. Please run again and enter numerical values.")
        return

    # 3. Main Training Loop
    epoch = 1
    total_error = float('inf')

    while total_error > error_threshold and epoch <= max_epochs:
        print(f"\n{'='*20} EPOCH {epoch} {'='*20}")
        
        # --- FORWARD PASS ---
        print("\n--- FORWARD PASS ---")
        y_hid = [0] * n_hid
        for j in range(n_hid):
            terms = [f"({x[i]} * {v[i][j]:.5g})" for i in range(n_in)]
            net_val = sum(x[i] * v[i][j] for i in range(n_in))
            print(f"net_hidden_{j+1} = {' + '.join(terms)} = {net_val:.5f}")
            
            y_hid[j] = sigmoid(net_val)
            print(f"y_hidden_{j+1}   = 1 / (1 + e^-{net_val:.5f}) = {y_hid[j]:.5f}\n")

        y_out = [0] * n_out
        for k in range(n_out):
            terms = [f"({y_hid[j]:.5f} * {w[j][k]:.5g})" for j in range(n_hid)]
            net_val = sum(y_hid[j] * w[j][k] for j in range(n_hid))
            print(f"net_output_{k+1} = {' + '.join(terms)} = {net_val:.5f}")
            
            y_out[k] = sigmoid(net_val)
            print(f"y_output_{k+1}   = 1 / (1 + e^-{net_val:.5f}) = {y_out[k]:.5f}\n")

        # --- ERROR CALCULATION ---
        error_terms = [f"({d[k]} - {y_out[k]:.5f})^2" for k in range(n_out)]
        total_error = 0.5 * sum((d[k] - y_out[k])**2 for k in range(n_out))
        print("--- NETWORK ERROR ---")
        print(f"E = 0.5 * [ {' + '.join(error_terms)} ] = {total_error:.6f}")
        
        if total_error <= error_threshold:
            print(f"\n[SUCCESS] Network Error ({total_error:.6f}) is below the threshold ({error_threshold}).")
            print(f"Training stopped after {epoch} epochs.")
            break

        # --- BACKWARD PASS ---
        print("\n--- BACKWARD PASS (Error Gradients) ---")
        delta_out = [0] * n_out
        for k in range(n_out):
            delta_out[k] = (d[k] - y_out[k]) * y_out[k] * (1 - y_out[k])
            print(f"delta_output_{k+1} = ({d[k]} - {y_out[k]:.5f}) * {y_out[k]:.5f} * (1 - {y_out[k]:.5f}) = {delta_out[k]:.6f}")
        print()

        delta_hid = [0] * n_hid
        for j in range(n_hid):
            sum_deltas = sum(delta_out[k] * w[j][k] for k in range(n_out))
            terms = [f"({delta_out[k]:.6f} * {w[j][k]:.5g})" for k in range(n_out)]
            
            delta_hid[j] = sum_deltas * y_hid[j] * (1 - y_hid[j])
            print(f"delta_hidden_{j+1} = [{' + '.join(terms)}] * {y_hid[j]:.5f} * (1 - {y_hid[j]:.5f}) = {delta_hid[j]:.6f}")
        print()

        # --- WEIGHT UPDATES ---
        print("--- WEIGHT UPDATES ---")
        for j in range(n_hid):
            for k in range(n_out):
                w_new = w[j][k] + (eta * delta_out[k] * y_hid[j])
                print(f"w_{j+1}{k+1}(new) = {w[j][k]:.5g} + ({eta} * {delta_out[k]:.6f} * {y_hid[j]:.5f}) = {w_new:.5f}")
                w[j][k] = w_new 
        print()

        for i in range(n_in):
            for j in range(n_hid):
                v_new = v[i][j] + (eta * delta_hid[j] * x[i])
                print(f"v_{i+1}{j+1}(new) = {v[i][j]:.5g} + ({eta} * {delta_hid[j]:.6f} * {x[i]:.5g}) = {v_new:.5f}")
                v[i][j] = v_new
        print()
        
        epoch += 1

    if epoch > max_epochs:
        print(f"\n[!] Training stopped: Maximum epochs ({max_epochs}) reached before hitting error threshold.")
        print(f"Final Network Error: {total_error:.6f}")

if __name__ == "__main__":
    main()