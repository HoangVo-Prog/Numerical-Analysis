import numpy as np
import matplotlib.pyplot as plt

def get_g(x):
    return -1

def get_h(x):
    return -2 

def get_f(x):
    return np.cos(x)

def exact_solution(x):
    return (-3/10 * np.cos(x) - 1/10 * np.sin(x))

a = 0
b = np.pi/2
N = 4 
alpha = -0.3
beta = -0.1

h = (b - a) / N

X = np.zeros(N + 1)
for i in range(N + 1):
    X[i] = a + i * h
    
G = np.zeros(N - 1)
H = np.zeros(N - 1)
F = np.zeros(N - 1)

for i in range(1, N):
    G[i - 1] = get_g(X[i])
    H[i - 1] = get_h(X[i])
    F[i - 1] = get_f(X[i])
    
Y = np.zeros(N + 1)
Y[0] = alpha
Y[N] = beta

a_sub = np.zeros(N - 1)   
b_diag = np.zeros(N - 1)  
c_sup = np.zeros(N - 1)   

for i in range(1, N):
    a_sub[i - 1] = 1.0/(h*h) - G[i - 1]/(2*h)
    b_diag[i - 1] = -2.0/(h*h) + H[i - 1]
    c_sup[i - 1] = 1.0/(h*h) + G[i - 1]/(2*h)

A = np.zeros((N - 1, N - 1))
B = np.zeros(N - 1)

for i in range(1, N):
    row = i - 1
    if i == 1:
        # First interior node: has main and upper diag, plus left boundary in RHS
        A[row, row] = b_diag[row]
        A[row, row + 1] = c_sup[row]          
        B[row] = F[row] - a_sub[row] * Y[0]   
    elif i == N - 1:
        # Last interior node: has sub and main diag, plus right boundary in RHS
        A[row, row - 1] = a_sub[row]
        A[row, row]     = b_diag[row]
        B[row] = F[row] - c_sup[row] * Y[N]   
    else:
        # Middle rows: full tri-diagonal
        A[row, row - 1] = a_sub[row]
        A[row, row]     = b_diag[row]
        A[row, row + 1] = c_sup[row]
        B[row] = F[row]


Y[1:N] = np.linalg.solve(A, B)

Y_exact = exact_solution(X)

print("Y:", Y[:10])
print("Y_exact:", Y_exact[:10])

print("Sai số tuyệt đối tại các điểm lưới:", np.abs(Y - Y_exact))
print("Sai số tuyệt đối lớn nhất:", np.max(np.abs(Y - Y_exact)))

plt.figure(figsize=(8, 5))
plt.plot(X, Y, 'o-', label='Cauchy Approximation')
plt.plot(X, Y_exact, 'r--', label='Exact Solution')
plt.title("So sánh nghiệm xấp xỉ và nghiệm chính xác")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()
