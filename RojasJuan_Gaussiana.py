#!/usr/bin/env python3
import numpy as np
N=3
#A=(np.random.random((N,N))*10.0)-5.0
A = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
print(A)
B=(np.random.random((N,1))*10.0)-5.0
print(B)
A_ = np.copy(A)
B_ = np.copy(B)

def laser_ray(M,b,i):
    b[i] = 1/M[i,i]*b[i]
    M[i] = 1/M[i,i]*M[i]
    for j in range(i+1,N):
        b[j] = -M[j,i]*b[i] + b[j]
        M[j] = -M[j,i]*M[i] + M[j]

    return M,b

def row_echelon_form(M, b):
    for i in range(N):
        if M[i,i] != 0:
            M,b = laser_ray(M,b,i)
        else:
            for j in range(i+1, N):
                if M[j,i] != 0:
                    b[i], b[j] = b[j], b[i]
                    M[i], M[j] = M[j], M[i]
                    M,b = laser_ray(M,b,i)
                    print(M[i,i])
                    break
            if j == N:
                raise ValueError('La matriz A no es invertible.')
    return M,b

def gauss_el(M,b):
    x = np.empty(N)
    M,b = row_echelon_form(M,b)
    print(M,b)
    x[-1] = b[-1]
    for n in range(2,N+1):
        x[-n] = -np.dot(M[-n][-n+1:], x[-n+1:]) + b[-n]
    return x
print(gauss_el(A,B), np.linalg.solve(A_,B_))
