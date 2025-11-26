import numpy as np
import math
from matplotlib import pyplot as plt

def afficher_matrice(matrice):
    if not matrice or not isinstance(matrice, list):
        print("La matrice est vide ou invalide.")
        return

    col_widths = [
        max(len(str(matrice[i][j])) for i in range(len(matrice))) 
        for j in range(len(matrice[0]))
    ]

    for ligne in matrice:
        print(" | ".join(f"{str(val):<{col_widths[i]}}" for i, val in enumerate(ligne)))

def get_q(d):
    b_d=d%10
    a_d=(d-b_d)/10
    
    if b_d==3:
        n=a_d*4
        q=(3*n)/4+1
        
    if b_d==7:
        n=a_d*4+1
        q=(7*(n-1))/4+5
        
    if b_d==9:
        n=a_d*4+2
        q=(n-2)/4+1
        
    if b_d==1:
        n=(a_d-1)*4+3
        q=(9*(n-3))/4+10
    
    return q

def get_A(d,j_max):
    A=[]
    L=[]
        
    q=get_q(d)
    
    for j in range(1,j_max+1,1):
        for i in range(1,11,1):
            #si m est dans D
            #if i in [2,4,8,10]:
                L.append(j*d+(1-i)*q)
        A.append(L)
        L=[]
    return [list(row) for row in zip(*A)]

def get_Km(d,j_max):
    Km=[]
    L=[]
    #10j+C[3,7,9,1]t*(1-i)
    
    if (d-3)%10==0:
        b_d=3
        
    if (d-7)%10==0:
        b_d=7  
        
    if (d-9)%10==0:
        b_d=1
        
    if (d-1)%10==0:
        b_d=9
    
    for j in range(1,j_max+1,1):
        for i in range(1,11,1):
            #si m est dans D
            #if i in [2,4,8,10]:
                L.append(10*j+b_d*(1-i))
        Km.append(L)
        L=[]
    return [list(row) for row in zip(*Km)]

def get_B(jmax):
    result=[]
    L=[]
    for j in range(0,jmax,1):
        for i in range(0,10):
            L.append(j)
        result.append(L)
        L=[]
        
    return result

def get_J(jmax):
    result=[]
    L=[]
    for j in range(0,jmax,1):
        for i in range(0,10):
            L.append(i+1)
        result.append(L)
        L=[]
        
    return result

def mult_mat_with_scal(scalar,matrix):
    return [[element * scalar for element in row] for row in matrix]

def add_mat_with_vect(vect,matrix):
    if len(matrix) != len(vect):
        raise ValueError("Le nombre de scalaires doit correspondre au nombre de lignes de la matrice.")

    return [[element + vect[i] for element in row] for i, row in enumerate(matrix)]

def add_matrices(matrix1, matrix2):

    # Vérification des dimensions
    if len(matrix1) != len(matrix2) or any(len(row1) != len(row2) for row1, row2 in zip(matrix1, matrix2)):
        raise ValueError("Les matrices doivent avoir les mêmes dimensions pour être additionnées.")

    # Addition des matrices
    result = []
    for row1, row2 in zip(matrix1, matrix2):
        result.append([a + b for a, b in zip(row1, row2)])

    return result

def multiply_matrices(matrix1, matrix2):

    # Vérification des dimensions
    if len(matrix1[0]) != len(matrix2):
        raise ValueError("Le nombre de colonnes de la première matrice doit être égal au nombre de lignes de la deuxième matrice.")

    # Multiplication des matrices
    result = []
    for row in matrix1:
        result_row = []
        for col in zip(*matrix2):
            result_row.append(sum(a * b for a, b in zip(row, col)))
        result.append(result_row)

    return result

def get_M(A):
    A=mult_mat_with_scal(10,A)
    M=add_mat_with_vect([0,1,2,3,4,5,6,7,8,9],A)
    return M

def get_alpha(d):
    b_d=d%10
    a_d=(d-b_d)/10
    
    if b_d==3:
        n=a_d*4
        alpha_n=n/4
        
    if b_d==7:
        n=a_d*4+1
        alpha_n=(n-1)/4
        
    if b_d==9:
        n=a_d*4+2
        alpha_n=(n-2)/4
        
    if b_d==1:
        n=(a_d-1)*4+3
        alpha_n=(n-3)/4
        
    return alpha_n

def get_A0(d,jmax):
    b_d=d%10
    
    if b_d==3:
        A0=get_A(3,jmax)
        
    if b_d==7:
        A0=get_A(7,jmax)
        
    if b_d==9:
        A0=get_A(9,jmax)
        
    if b_d==1:
        A0=get_A(11,jmax)
        
    return A0



d=9
j_max=10

A=get_A(d,j_max)
M=get_M(A)
K=get_Km(d,j_max)
afficher_matrice(M)
afficher_matrice(K)