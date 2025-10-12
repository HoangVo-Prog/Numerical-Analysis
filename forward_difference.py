import numpy as np
import matplotlib.pyplot as plt

def alpha_value(x):
    return x

def beta_value(x):
    return -x**3

def exact_solution(x):
    return x**2 + 2 - np.exp(x**2/2)

a = 0
b = 2
N = 50
theta = 1


h = (b - a) / N
X = np.linspace(a, b, N)
Y = np.zeros(N)
alpha = alpha_value(X) 
beta = beta_value(X)

Y[0] = theta
for i in range(N-1):
    Y[i+1] = Y[i] + h * (alpha[i] * Y[i] + beta[i])
    
print("X:", X)
print("Y:", Y)

# --- Nghiệm chính xác ---
Y_exact = exact_solution(X)

# --- Hiển thị ---
plt.figure(figsize=(8, 5))
plt.plot(X, Y, 'o-', label='Euler Approximation')
plt.plot(X, Y_exact, 'r--', label='Exact Solution')
plt.title("So sánh nghiệm xấp xỉ và nghiệm chính xác")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()