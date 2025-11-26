import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.optimize import fsolve

#Générée dans la nouvelle définition de n
def plot_m(matrice):
    """
    Afficher les données brutes d'une matrice dans la console 
    
    """
    if not matrice or not isinstance(matrice, list):
        print("La matrice est vide ou invalide.")
        return

    col_widths = [
        max(len(str(matrice[i][j])) for i in range(len(matrice))) 
        for j in range(len(matrice[0]))
    ]

    for ligne in matrice:
        print(" | ".join(f"{str(val):<{col_widths[i]}}" for i, val in enumerate(ligne)))
    print("\n")

def display_matrix(matrix):
    """
    Afficher les valeurs d'une matrice avec une coloration pour un rendu plus visible
    
    """
    matrix = np.array(matrix)  # Conversion en tableau NumPy pour faciliter l'affichage
    size=matrix.shape[0]
    plt.figure(figsize=(size, 10))
    plt.imshow(matrix, cmap="coolwarm", origin="upper")  # 'upper' garde i descendant
    
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, str(matrix[i, j]), ha='center', va='center', color='black')
    
    plt.xticks(np.arange(size), labels=np.arange(1, size + 1))
    plt.yticks(np.arange(10), labels=np.arange(1, 11)[::-1])  # Axe i descendant
    
    plt.xlabel("j ")
    plt.ylabel("i ")
    plt.title("Affichage de la matrice Km")
    plt.colorbar(label="Valeurs de Km")
    
    plt.show()

def display_matrices(matrices):
    """
    Permet d'affficher l'espace 3D obtenu si l'on superpose des matrices suivant un axe p
    <!> Quelques modifications sont encore à prévoir
    
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    size = matrices[0].shape[0]
    j_max = matrices[0].shape[1]

    # Trouver le maximum global sur toutes les matrices
    global_max = max(matrix.max() for matrix in matrices)

    for p, matrix in enumerate(matrices):
        matrix = np.array(matrix)
        Y, X = np.meshgrid(np.arange(j_max), np.arange(size)[::1])
        Z = np.full_like(X, p)

        # Normalisation par rapport au maximum global
        face_colors = plt.cm.coolwarm(matrix / global_max)

        ax.plot_surface(X, Y, Z, facecolors=face_colors, rstride=1, cstride=1, alpha=0.8)

    ax.set_xlabel("axe i")
    ax.set_ylabel("axe j")
    ax.set_zlabel("axe p")
    ax.set_title("Affichage des matrices A dans l'espace")

    plt.show()



def mult_mat_with_scal(scalar,matrix):
    """
    Permet de multiplier des matrices en L[][] avec un scalaire sans passer par numpy
    
    """
    return [[element * scalar for element in row] for row in matrix]

def add_mat_with_vect(vect,matrix):
    """
    Permet d'additionner des matrices en L[][] avec un vecteur (en ajoutant la valeur de sa ligne i a tous les termes de la ligne i de la matrice) 
    sans passer par numpy

    
    """
    if len(matrix) != len(vect):
        raise ValueError("Le nombre de scalaires doit correspondre au nombre de lignes de la matrice.")

    return [[element + vect[i] for element in row] for i, row in enumerate(matrix)]

def add_matrices(matrix1, matrix2):
    
    """
    Permet d'additionner des matrices en L[][] entre elles sans passer par numpy

    """

    # Vérification des dimensions
    if len(matrix1) != len(matrix2) or any(len(row1) != len(row2) for row1, row2 in zip(matrix1, matrix2)):
        raise ValueError("Les matrices doivent avoir les mêmes dimensions pour être additionnées.")

    # Addition des matrices
    result = []
    for row1, row2 in zip(matrix1, matrix2):
        result.append([a + b for a, b in zip(row1, row2)])

    return result

def multiply_matrices(matrix1, matrix2):
    
    """
    Permet de multiplier des matrices en L[][] entre elles sans passer par numpy

    """

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

def get_q(d):
    """
    Retourne q(d), le nombre q associé au diviseur d dans D
    
    """
    b_d=d%10
    a_d=(d-b_d)/10
    
    if b_d==1:
        n=a_d*4
        q=(9*n)/4+1
    
    
    if b_d==3:
        n=a_d*4+1
        q=(3*(n-1))/4+1
        
    if b_d==7:
        n=a_d*4+2
        q=(7*(n-2))/4+5
        
    if b_d==9:
        n=a_d*4+3
        q=(n-3)/4+1    
    return q

def get_d(n):
    """
    Retourne d_n étant le diviseur d associé au nombre n (si l'on numérote les éléments de D dans l'ordre croissant)
    
    """
    
    n4=n%4
    p=(n-n4)/4
    if n4==0:
        d0=1
    if n4==1:
        d0=3
    if n4==2:
        d0=7
    if n4==3:
        d0=9
    
    d=10*p+d0
    return(d)

def get_c(d0):
    """
    Retourne c en considérant la condition delta choisie (repérée par d0 placé en argument)
    
    """
    if d0==1:
        return 9
    if d0==3:
        return 3
    if d0==7:
        return 7
    if d0==9:
        return 1
    
def get_q0(d0):
    """
    Retourne q0 en considérant la condition delta choisie (repérée par d0 placé en argument)
    
    """
    if d0==1:
        return 1
    if d0==3:
        return 1
    if d0==7:
        return 5
    if d0==9:
        return 1


def get_A(d,j_max):
    """
    Retourne la matrice A_n associé au d_n placé en argument avec 10 lignes et un nombre j_max de colonnes 
    
    """
    A=[]
    L=[]
        
    q=get_q(d)
    for j in range(1,j_max+1,1):
        for i in range(1,11,1):
                L.append(j*d+(1-i)*q)
        A.append(L)
        L=[]
    return [list(row) for row in zip(*A)]

def get_Km(d,j_max):
    """
    Retourne la matrice K_n associé au d_n placé en argument avec 10 lignes et un nombre j_max de colonnes 
    Possible de remplacer d par d0 étant donné la nature de K_m
    
    """

    Km=[]
    L=[]
    
    if (d-1)%10==0:
        b_d=9
    
    if (d-3)%10==0:
        b_d=3
        
    if (d-7)%10==0:
        b_d=7  
        
    if (d-9)%10==0:
        b_d=1
            
    for j in range(1,j_max+1,1):
        for i in range(1,11,1):
                L.append(10*j+b_d*(1-i))
        Km.append(L)
        L=[]
    return [list(row) for row in zip(*Km)]

def get_B(jmax):
    """
    Retourne la matrice B tq M= 10A + B avec 10 lignes et un nombre j_max de colonnes 

    """

    result=[]
    L=[]
    for j in range(0,jmax,1):
        for i in range(0,10):
            L.append(j)
        result.append(L)
        L=[]
        
    return result

def get_J(jmax):
    """
    Retourne la matrice J tq A+qB=dJ avec 10 lignes et un nombre j_max de colonnes 

    """
    result=[]
    L=[]
    for j in range(0,jmax,1):
        for i in range(0,10):
            L.append(i+1)
        result.append(L)
        L=[]
        
    return result

def get_M(A):
    """
    Retourne la matrice M tq M= 10A + B avec 10 lignes et un nombre j_max de colonnes 

    """
    A=mult_mat_with_scal(10,A)
    M=add_mat_with_vect([0,1,2,3,4,5,6,7,8,9],A)
    return M

def get_A0(d,jmax):
    """
    Retourne la matrice A si n vaut 0
    d peut être remplacé par d0 étant donné la nature de A0

    """
    b_d=d%10
    
    if b_d==1:
        A0=get_A(1,jmax)
    
    if b_d==3:
        A0=get_A(3,jmax)
        
    if b_d==7:
        A0=get_A(7,jmax)
        
    if b_d==9:
        A0=get_A(9,jmax)
        

        
    return A0


def get_A_p(p,d0,j_max):
    """
    Retourne la matrice A si p est un rationnel

    """
    A=[]
    L=[]
    Km=get_Km(d0,j_max)
    A0=get_A0(d0,j_max)
    for j in range(0,j_max,1):
        for i in range(0,10,1):
                L.append(p*Km[i][j]+A0[i][j])
        A.append(L)
        L=[]
    return [list(row) for row in zip(*A)]

def get_M_p(p,d0,j_max):
    """
    Retourne la matrice M si p est un rationnel
    
    """
    A_p=get_A_p(p,d0,j_max)
    B=get_B(j_max)
    
    T=mult_mat_with_scal(10,A_p)
    
    M_p= add_matrices(T,B)
    
    return M_p

def get_d_p(p,d0):
    """
    Retourne d si p est un rationnel
    
    """
    return 10*p+d0

def get_q_p(p,d0):
    """
    Retourne d si p est un rationnel en veillant à transmettre la bonne condition delta (repérée par d0)
    
    """
    c=get_c(d0)
    q0=get_q0(d0)
    return c*p+q0

def plot_evol_d(p_min, p_max,nb_points):
    """
    Trace sur une même courbe la fonction d(p)=10p+d0 pour les quatre valeurs possibles de d0 
    
    """
    p = np.arange(p_min, p_max ,nb_points)  # Génère les valeurs de p
    d0_values = [1, 3, 7, 9]  # Valeurs de d0
    
    plt.figure(figsize=(8, 6))
    
    for d0 in d0_values:
        d_values = 10 * p + d0
        plt.plot(p, d_values, label=f'd0 = {d0}')
    
    plt.xlabel('p')
    plt.ylabel('d')
    plt.title('Évolution de d = 10p + d0 en fonction de p')
    plt.legend()
    plt.grid()
    plt.show()
    
def plot_evol_q(p_min, p_max,nb_points):
    """
    Trace sur une même courbe la fonction q(p)=cp+q0 pour les quatre valeurs possibles de d0 
    
    """
    p = np.linspace(p_min, p_max, nb_points)  # Génère les valeurs de p
    q0_values = [1, 1, 5, 1]  # Valeurs de q0
    c_values = [9, 3, 7, 1] # Valeurs de c
    d0_values = [1, 3, 7, 9] # Valeurs de d0
    
    plt.figure(figsize=(8, 6))
    
    for q0_idx in range (len(q0_values)):
        q0=q0_values[q0_idx]
        c=c_values[q0_idx]
        d_values = c * p + q0
        plt.plot(p, d_values, label=f'd0 = {d0_values[q0_idx]}')
    
    plt.xlabel('p')
    plt.ylabel('d')
    plt.title('Évolution de d = 10p + d0 en fonction de p')
    plt.legend()
    plt.grid()
    plt.show()

def plot_evol_d_and_q(p_min,p_max,nb_points):
    """
    Permet de trouver les points d'intersections des fonctions d(p) et q(p) pour toutes les conditions
        
    Puis trace sur une même courbe ces fonctions

    """
    I={} #dico permettant d'afficher les points d'intersection
    plt.figure(figsize=(8, 6))
    
    def find_intersections(f, g, p_min,p_max):
 
        # Définir la fonction d'écart entre f(x) et g(x)
        def intersection(x):
            return f(x) - g(x)

        # Trouver les points d'intersection
        x_vals = np.linspace(p_min, p_max, 1000)
        intersections = []

        for i in range(1, len(x_vals)):
            # Si le signe de l'écart change entre deux points consécutifs, il y a une intersection
            if intersection(x_vals[i-1]) * intersection(x_vals[i]) < 0:
                # Résoudre l'équation f(x) = g(x) dans cet intervalle
                sol = fsolve(intersection, (x_vals[i-1], x_vals[i]))
                intersections.append(sol[0])

        return [np.round(intersections[0], 3),np.round(f(intersections[0]), 3)]
    
    d0 = [1, 3, 7, 9]  # Valeurs de d0
    q0 = [1, 1, 5, 1]  # Valeurs de q0
    c = [9, 3, 7, 1]  # Valeurs de c
    colors = {1: 'blue', 3: 'green', 7: 'red', 9: 'purple'} 
    p = np.linspace(p_min, p_max + 1,nb_points)  # Génère les valeurs de p
    
    for d0_idx in range(len(d0)):
        
        def d(p):
            return 10*p+d0[d0_idx] 
        
        def q(p):
            return c[d0_idx] *p+q0[d0_idx]
        
        pt_inter=find_intersections(d,q,p_min,p_max)
        I[colors[d0[d0_idx]]]=pt_inter
        plt.plot(p, 10 * p + d0[d0_idx], label=f'd(p) d0 = {d0[d0_idx]}', color=colors[d0[d0_idx]])
        plt.plot(p, c[d0_idx] * p + q0[d0_idx], label=f'q(p) d0 = {d0[d0_idx]}', color=colors[d0[d0_idx]], linestyle='dashed')
    
    for key in I.keys():
        print("{} : {}\n".format(key,I[key]))
        
    plt.xlabel('p')
    plt.ylabel('Valeur')
    plt.title("Évolution de d(p) et q(p) en fonction de p")
    plt.legend()
    plt.grid()
    plt.show()
    
def get_p_inter(d0):
    """
    Permet d'avoir p_inter de la condition désirée
    
    """
    q0=get_q0(d0)
    c=get_c(d0)
    
    return np.round((q0-d0)/(10-c),3)

def get_d_inter(d0):
    """
    Permet d'avoir d_inter de la condition désirée
    
    """
    q0=get_q0(d0)
    c=get_c(d0)
    
    return np.round(10*((q0-d0)/(10-c)) + d0,3)

def get_q_inter(d0):
    """
    Permet d'avoir q_inter de la condition désirée
    
    """
    q0=get_q0(d0)
    c=get_c(d0)
    
    return np.round(c*((q0-d0)/(10-c)) + q0,3)
    
def get_A_inter(d0,j_max):
    """
    Permet d'avoir la matrice A_inter dans la condition désirée
    
    """
    d_inter=get_d_inter(d0)
    A=[]
    L=[]
    for j in range(1,j_max+1,1):
        for i in range(1,11,1):
                L.append(np.round((j-i+1)*d_inter,3))
        A.append(L)
        L=[]
    return [list(row) for row in zip(*A)]

def get_M_inter(d0,j_max):
    """
    Permet d'avoir la matrice M_inter dans la condition désirée
    
    """
    A_inter=get_A_inter(d0,j_max)
    B=get_B(j_max)
    
    T=mult_mat_with_scal(10,A_inter)
    
    M_inter= add_matrices(T,B)
    
    return M_inter

def plot_center_of_all():
    """
    Affiche une construction géométrique basée sur le cercle circonscrit des centres des cercles circonscrits des triangles formés par les intersections
    
    """
    def circumcircle(s1, s2, s3, ax):
        d0_values = [1, 3, 7, 9, 1, 3, 9]
        P = [float(get_p_inter(d0)) for d0 in d0_values]
        Q = [float(get_q_inter(d0)) for d0 in d0_values]

        A = np.array([P[s1], Q[s1]], dtype=float)
        B = np.array([P[s2], Q[s2]], dtype=float)
        C = np.array([P[s3], Q[s3]], dtype=float)

        det = (B[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (B[1] - A[1])
        if abs(det) < 1e-6:
            print("Erreur : les points sont alignés ! Impossible de trouver un cercle circonscrit.")
            return None

        D = 2 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))

        Ux = ((A[0]**2 + A[1]**2) * (B[1] - C[1]) +
            (B[0]**2 + B[1]**2) * (C[1] - A[1]) +
            (C[0]**2 + C[1]**2) * (A[1] - B[1])) / D

        Uy = ((A[0]**2 + A[1]**2) * (C[0] - B[0]) +
            (B[0]**2 + B[1]**2) * (A[0] - C[0]) +
            (C[0]**2 + C[1]**2) * (B[0] - A[0])) / D

        circumcenter = np.array([Ux, Uy])
        radius = np.linalg.norm(circumcenter - A)

        ax.add_patch(plt.Circle(circumcenter, radius, color='b', fill=False, linestyle='dashed'))
        ax.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], 'ro-')

        ax.scatter([circumcenter[0]], [circumcenter[1]], color='green')

        return circumcenter


    fig, ax = plt.subplots()
    
    print("----------cercle 1---------------\n")
    c1 = circumcircle(0, 1, 3, ax)
    print(f"-> centre : {c1}\n")
    
    print("----------cercle 2---------------\n")
    c2 = circumcircle(0, 1, 2, ax)
    print(f"-> centre : {c2}\n")

    print("----------cercle 3---------------\n")
    c3 = circumcircle(1, 2, 3, ax)
    print(f"-> centre : {c3}\n")


    if c1 is None or c2 is None or c3 is None:
        print("Impossible de construire le dernier cercle.")
        return

    print("----------cercle total---------------\n")

    det = (c2[0] - c1[0]) * (c3[1] - c1[1]) - (c3[0] - c1[0]) * (c2[1] - c1[1])
    if abs(det) < 1e-6:
        print("Erreur : les points sont alignés ! Impossible de trouver un cercle circonscrit.")
        return

    D = 2 * (c1[0] * (c2[1] - c3[1]) + c2[0] * (c3[1] - c1[1]) + c3[0] * (c1[1] - c2[1]))

    Ux = ((c1[0]**2 + c1[1]**2) * (c2[1] - c3[1]) +
          (c2[0]**2 + c2[1]**2) * (c3[1] - c1[1]) +
          (c3[0]**2 + c3[1]**2) * (c1[1] - c2[1])) / D

    Uy = ((c1[0]**2 + c1[1]**2) * (c3[0] - c2[0]) +
          (c2[0]**2 + c2[1]**2) * (c1[0] - c3[0]) +
          (c3[0]**2 + c3[1]**2) * (c2[0] - c1[0])) / D

    circumcenter = np.array([Ux, Uy])
    radius = np.linalg.norm(circumcenter - c1)

    ax.add_patch(plt.Circle(circumcenter, radius, color='orange', fill=False, linestyle='dashed'))
    ax.plot([c1[0], c2[0], c3[0], c1[0]], [c1[1], c2[1], c3[1], c1[1]], '-p',color='purple', label="Triangle des centres")

    ax.scatter([circumcenter[0]], [circumcenter[1]], color='green', label="Centre final")

    print(f"-> centre : {circumcenter }\n")


    ax.set_aspect('equal')
    plt.legend()
    plt.grid()
    plt.show()
    
def get_k_family(d,m):
    q=get_q(d)
    L_K=[]
    for j in range (len(str(m))):
        s=0
        t=0
        for i in range (len(str(m))-j):
            s+=10**(len(str(m))-i-1-j)*int(str(m)[i])
        for i in range(j):
            t+=q**(j-i)*int(str(m)[-i-1])
            
        r=s+t
        k=r/d
        print(f"{s} + {t} = {r} = {k} * {d}")
        L_K.append(k)
    print(L_K)
