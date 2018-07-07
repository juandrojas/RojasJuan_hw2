#!/usr/bin/env python3
#importamos paquetes
import numpy as np
N=3
A=(np.random.random((N,N))*10.0)-5.0
#A = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]) #este testea el cambio de filas
#A = np.array([[1, 0, 0], [0, 0, 1], [0, 0, 0]]) #este testea si se detecta la no-invertibilidad
print("La matriz de coeficientes A es:",A)
B=(np.random.random((N,1))*10.0)-5.0
print("El vector B es:",B)

#estas copias cumplen el propósito de testear con el método np.linalg.solve
A_ = np.copy(A)
B_ = np.copy(B)

#'laser_ray' elimina las entradas por debajo del pivote 'M[i,i]' del sistema 'M'x = 'b' como un rayo laser.
def laser_ray(M,b,i):
    b[i] = 1/M[i,i]*b[i]
    M[i] = 1/M[i,i]*M[i]
    for j in range(i+1,N):
        b[j] = -M[j,i]*b[i] + b[j]
        M[j] = -M[j,i]*M[i] + M[j]

    return M,b
#retorna la forma reducida por renglones del sistema 'M'x = 'b'. Lanza una excepción si la matriz A no es invertible; sin embargo, calcula su forma reducida de todas formas
def row_echelon_form(M, b):
    for i in range(N):
        if M[i,i] != 0:
            M,b = laser_ray(M,b,i)
        else:
            for j in range(i+1, N):
                if M[j,i] != 0:
                    M[[i,j]] = M[[j,i]]
                    b[[i,j]] = b[[j,i]]
                    M,b = laser_ray(M,b,i)
                    break
            if M[i,i] == 0:
                raise ValueError('La matriz A no es invertible.')
                continue
    return M,b

#retorna la solucion al sistema 'M'x = 'b'
def gauss_el(M,b):
    x = np.empty([N, 1])
    M,b = row_echelon_form(M,b)
    x[-1] = b[-1]
    for n in range(2,N+1):
        x[-n] = -np.dot(M[-n][-n+1:], x[-n+1:]) + b[-n]
    return x

#testing
print("El vector solución del sistema es:",gauss_el(A,B))
print(np.linalg.solve(A,B))
