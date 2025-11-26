import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Définition des constantes
c = 1  
d0 = 9  
q0 = 1  

# Définition des plages d'entiers pour i, j et p
i_values = np.arange(-10, 11, 1)  # i varie de -10 à 10 (entiers)
j_values = np.arange(-10, 11, 1)  # j varie de -10 à 10 (entiers)
p_values = np.arange(-20, 21, 1)  # p varie de -10 à 10 (entiers)

# Création des grilles 3D
i_grid, j_grid, p_grid = np.meshgrid(i_values, j_values, p_values, indexing='ij')

# Calcul des valeurs de a
a_values = p_grid * (10 * j_grid + c * (1 - i_grid)) + j_grid * d0 + q0 * (1 - i_grid)

# Conversion en listes 1D pour scatter plot
i_flat = i_grid.flatten()
j_flat = j_grid.flatten()
p_flat = p_grid.flatten()
a_flat = a_values.flatten()

# Création de la figure 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter 3D avec couleurs basées sur a
sc = ax.scatter(i_flat, j_flat, p_flat, c=a_flat, cmap='viridis', marker='o')



# Ajouter une barre de couleur
cbar = plt.colorbar(sc, ax=ax, shrink=0.5)
cbar.set_label('Valeur de a')

# Labels et titre
ax.set_xlabel('i')
ax.set_ylabel('j')
ax.set_zlabel('p')
ax.set_title('Variation de a en fonction de (i, j, p) - j inversé')

plt.show()
