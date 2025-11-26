
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
P=[2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293]

def decompo(n):
    """
    Décompose un entier n en facteurs premiers.

    Paramètre:
    n (int): L'entier à décomposer.

    Retourne:
    dict: Un dictionnaire où les clés sont les facteurs premiers et les valeurs sont leurs puissances respectives.
    """
    facteurs = {}
    diviseur = 2

    while n > 1:
        while (n % diviseur) == 0:
            if diviseur in facteurs:
                facteurs[diviseur] += 1
            else:
                facteurs[diviseur] = 1
            n //= diviseur
        diviseur += 1

    return facteurs

def avoir_L(facteurs):
    Pui=[]
    L1=[]
    L2=[]
    for p in facteurs:
        Pui.append(facteurs[p])
    D=list(facteurs)
    idx=0
    while P[idx]!=D[-1]:
        idx+=1
    idx+=1
    a=0
    for i in range(idx):
        if P[i] in D:
            L1.append(Pui[a])
            a+=1
        else:
            L1.append(0)
    for j in range(1,len(L1)+1,1):
        L2.append(L1[-j])
    return L2

def trouver_m(L):
    prod=1
    idx=len(L)-1
    for i in range(0,len(L),1):
        p=P[idx-i]
        prod*=p**L[i]
    return prod

def zero_padding(L,n):
    if len(L)==n:
        return L
    else:
        L.append(0)
        L= [L[-1]] + L[:-1]
        return zero_padding(L,n)



def plot_ndimensional_data(data, method='PCA', dimensions=2):
    if dimensions not in [2, 3]:
        raise ValueError("La dimension cible doit être 2 ou 3")

    if method == 'PCA':
        reducer = PCA(n_components=dimensions)
    elif method == 't-SNE':
        reducer = TSNE(n_components=dimensions)
    else:
        raise ValueError("La méthode doit être 'PCA' ou 't-SNE'")

    reduced_data = reducer.fit_transform(data)
    
    if dimensions == 2:
        plt.figure()
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
        for i in range(len(data)):
            plt.text(reduced_data[i, 0], reduced_data[i, 1], str(i+2), fontsize=9, ha='right')
        plt.xlabel("Composante 1")
        plt.ylabel("Composante 2")
        plt.title(f"Projection 2D utilisant {method}")
        plt.show()
    elif dimensions == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2])
        for i in range(len(data)):
            ax.text(reduced_data[i, 0], reduced_data[i, 1], reduced_data[i, 2], str(i), fontsize=9, ha='right')
        ax.set_xlabel("Composante 1")
        ax.set_ylabel("Composante 2")
        ax.set_zlabel("Composante 3")
        plt.title(f"Projection 3D utilisant {method}")
        plt.show()


def voir_3D(px,py,pz,L):
    x=[]
    y=[]
    z=[]
    ix=0
    iy=0
    iz=0
    while P[ix]!=px:
        ix+=1
    ix+=1
    while P[iy]!=py:
        iy+=1
    iy+=1
    while P[iz]!=pz:
        iz+=1
    iz+=1
    
    for el in L :
        x.append(el[-ix])
        y.append(el[-iy])
        z.append(el[-iz])
        
    fig = plt.figure()

    # Ajouter une sous-figure 3D
    ax = fig.add_subplot(111, projection='3d')

    # Tracer les données
    ax.scatter(x, y, z, c='r', marker='o')
    
    for i in range(len(x)):
            ax.text(x[i], y[i], z[i], str(i+2), fontsize=9, ha='right')
    # Ajouter des étiquettes aux axes
    ax.set_xlabel(f'Axe U{px}')
    ax.set_ylabel(f'Axe U{py}')
    ax.set_zlabel(f'Axe U{pz}')

    # Afficher le graphique
    plt.show()




Decompo=[]

for i in range(2,89,1):
    m = i
    facteurs = decompo(m)
    L=avoir_L(facteurs)
    print(f"La décomposition en facteurs premiers de {m} est : {L}\n")
    Decompo.append(zero_padding(L,len(avoir_L(decompo(293)))))

voir_3D(23,37,47,Decompo)
