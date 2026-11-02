import numpy as np 

def sgn(net): return -1 if net < 0 else 1 

n = int(input("Enter number of input vectors: ")) 
r = int(input("Enter dimension of each vector: ")) 
X = [] 
print() 
for i in range(n): 
    X.append([float(x) for x in input(f"Enter x{i+1}: ").split()]) 
X = np.array(X) 
w = np.array([float(x) for x in input("Enter initial weights: ").split()]) 
d = np.array([int(x) for x in input("Enter desired responses: ").split()]) 
c = float(input("Enter c: ")) 
print() 
ws = [w.copy()] 
for i in range(n): 
    net = w.dot(X[i]) 
    y = sgn(net) 
    term = 2*c*X[i]*d[i] 
    if(y != d[i]): 
        w = w + term 
    ws.append(w.copy()) 
    print(f"Iteration {i+1}:") 
    print(f"Net = {net:.3f}") 
    print("y =", y) 
    print("W =", w) 
    print() 

print("All Weight Vectors:") 
for i, w in enumerate(ws): 
    print(f"W{i+1}: {w}") 

x_test = np.array([float(x) for x in input("\nEnter test data : ").split()]) 
net_test = w.dot(x_test) 
y_test = sgn(net_test) 
print("For test data :") 
print("Net = ", net_test) 
print("Predicted output = ", y_test) 