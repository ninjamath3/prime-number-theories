import numpy as np
import matplotlib.pyplot as plt

def generate_spiral(turns=10):
    theta = np.arange(0, turns * 2 * np.pi, np.pi / 2)  # Discrétisation par pas de 90°
    r =  theta / (np.pi / 2)  # Nouvelle formule pour augmentation tous les 90°
    return theta, r

def plot_spiral():
    theta, r = generate_spiral()
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, '-o', markersize=5)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.title("Spirale discrétisée avec incréments de 90°")
    plt.show()

plot_spiral()
