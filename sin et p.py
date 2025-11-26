import matplotlib.pyplot as plt
import sympy
import numpy as np

# Liste des nombres premiers
P = list(sympy.primerange(1, 500))

# Fonction sinus adaptée aux tableaux NumPy
def f_sin(p, X):
    Y = np.cos((np.pi * X) / p)
    return Y

# Valeurs pour X
X = np.linspace(0, 20, 10000)

# Nombre maximal de nombres premiers à utiliser
max_p =7

# Tracé des courbes
for i in range(max_p):
    nb_p = P[i]
    Y = f_sin(nb_p, X)
    plt.plot(X, Y, label=f'p={nb_p}')

# Affichage du graphique
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Courbes de sin(pi*X/p) pour différents nombres premiers p')
plt.legend()
plt.grid()
plt.show()
