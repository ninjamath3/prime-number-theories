import matplotlib.pyplot as plt
import sympy

# Liste des nombres premiers jusqu'à 110
P = list(sympy.primerange(1, 106))

def decompo(n):
    """
    Décompose un entier n en facteurs premiers.
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
        
    dico_decompo = facteurs
    dico = {}
    for p in P:
        if p in dico_decompo.keys():
            dico[p] = dico_decompo[p]
        else:
            dico[p] = 0
 
    return dico

def plot_curve(n):
    """
    Trace la courbe des facteurs premiers décomposés pour un entier n.
    """
    D = decompo(n)
    
    # Création des listes X et Y avec le point (0, 0)
    X = [1] + list(range(1, len(D) + 1))
    Y = [1] + list(D.values())

    plt.plot(X, Y, marker='o')  # Ajout de marqueurs pour visualiser les points
    plt.title(f"n = {n}")
    plt.xlabel("Facteurs premiers")
    plt.ylabel("Multiplicité")
    plt.xticks(X[1:], D.keys())  # Exclure le point 0 des étiquettes de l'axe des X
    plt.show(block=False)  # Affiche sans bloquer l'interface utilisateur
    return D


def add_to_class(class_index, n, classes):
    """
    Ajoute un nombre à une classe en utilisant l'index de la classe.
    """
    class_name = list(classes.keys())[class_index]
    classes[class_name].append(n)
    print(f"Ajouté {n} à la classe {class_name}")

def view_class_curve(class_index, classes):
    """
    Visualise une courbe pour un exemple de la classe donnée.
    """
    class_name = list(classes.keys())[class_index]
    if class_name in classes and len(classes[class_name]) > 0:
        n = classes[class_name][0]
        print(f"Visualisation d'une courbe pour la classe {class_name} avec n = {n}")
        plt.figure()  # Crée une nouvelle figure pour la visualisation de la classe
        plot_curve(n)
    else:
        print(f"Aucune courbe à visualiser pour la classe {class_name}")

def display_classes(classes):
    """
    Affiche les classes existantes avec leur index.
    """
    print("\nClasses existantes :")
    for index, class_name in enumerate(classes.keys()):
        print(f"{index} - {class_name}")

def edit_dictionary(classes):
    """
    Permet à l'utilisateur d'éditer le dictionnaire des classes.
    """
    while True:
        print("\nOptions d'édition :")
        print("1 - Ajouter une nouvelle classe")
        print("2 - Supprimer une classe")
        print("3 - Ajouter un élément à une classe")
        print("4 - Supprimer un élément d'une classe")
        print("5 - Quitter l'édition")
        
        option = input("Choisissez une option (1/2/3/4/5) : ")

        if option == "1":
            # Ajouter une nouvelle classe
            class_name = input("Entrez le nom de la nouvelle classe : ")
            if class_name in classes:
                print(f"La classe {class_name} existe déjà.")
            else:
                classes[class_name] = []  # Crée une nouvelle classe vide
                print(f"Classe {class_name} créée.")

        elif option == "2":
            # Supprimer une classe
            display_classes(classes)
            class_index = int(input("Entrez le numéro de la classe à supprimer : "))
            if 0 <= class_index < len(classes):
                class_name = list(classes.keys())[class_index]
                del classes[class_name]
                print(f"Classe {class_name} supprimée.")
            else:
                print("Numéro de classe invalide.")

        elif option == "3":
            # Ajouter un élément à une classe
            display_classes(classes)
            class_index = int(input("Entrez le numéro de la classe à laquelle ajouter un élément : "))
            if 0 <= class_index < len(classes):
                n = int(input("Entrez le nombre à ajouter : "))
                add_to_class(class_index, n, classes)
            else:
                print("Numéro de classe invalide.")

        elif option == "4":
            # Supprimer un élément d'une classe
            display_classes(classes)
            class_index = int(input("Entrez le numéro de la classe dont vous voulez supprimer un élément : "))
            if 0 <= class_index < len(classes):
                class_name = list(classes.keys())[class_index]
                if len(classes[class_name]) > 0:
                    print(f"Éléments dans la classe {class_name} : {classes[class_name]}")
                    n = int(input("Entrez le nombre à supprimer : "))
                    if n in classes[class_name]:
                        classes[class_name].remove(n)
                        print(f"Supprimé {n} de la classe {class_name}.")
                    else:
                        print(f"{n} n'est pas dans la classe {class_name}.")
                else:
                    print(f"Aucun élément dans la classe {class_name}.")
            else:
                print("Numéro de classe invalide.")

        elif option == "5":
            break  # Quitter l'édition

        else:
            print("Option invalide. Veuillez réessayer.")
def user_interface(N):
    """
    Interface utilisateur interactive pour gérer les classes et visualiser les courbes.
    """
    classes = {}
    current_n = 2  # Initialisation de n

    while current_n <= N:
        # Afficher le graphique pour n courant
        plt.clf()  # Efface le graphique précédent
        print(f"Affichage du graphique pour n = {current_n}")
        plot_curve(current_n)  # Affiche le graphique pour current_n

        action = ""
        while action not in {"1", "2", "3", "4", "5"}:
            print(f"\nOptions pour n = {current_n} :\n"
                  "1 - Créer une nouvelle classe\n"
                  "2 - Ajouter n à une classe\n"
                  "3 - Visualiser une courbe d'une classe\n"
                  "4 - Mettre à jour le dictionnaire\n"
                  "5 - Quitter\n")

            action = input("Choisissez une option (1/2/3/4/5) : ")

            if action == "1":
                # Créer une nouvelle classe
                class_name = input("Entrez le nom de la nouvelle classe : ")
                classes[class_name] = [current_n] 
                print(f"Classe {class_name} créée.")
                break  # Passer à n suivant

            elif action == "2":
                # Afficher les classes existantes et ajouter n à une classe existante
                display_classes(classes)
                class_index = int(input("Entrez le numéro de la classe à laquelle ajouter n : "))
                if 0 <= class_index < len(classes):
                    add_to_class(class_index, current_n, classes)
                    break  # Passer à n suivant
                else:
                    print("Numéro de classe invalide.")

            elif action == "3":
                # Afficher les classes existantes et visualiser une courbe SANS CHANGER current_n
                display_classes(classes)
                class_index = int(input("Entrez le numéro de la classe à visualiser : "))
                if 0 <= class_index < len(classes):
                    view_class_curve(class_index, classes)
                    input("Appuyez sur Entrée pour revenir au menu.")
                    plt.close()  # Fermer la courbe visualisée après la consultation
                else:
                    print("Numéro de classe invalide.")

            elif action == "4":
                # Mettre à jour le dictionnaire des classes sans changer current_n
                edit_dictionary(classes)

            elif action == "5":
                # Quitter le programme
                print("Fermeture du programme.")
                plt.close()  # Fermer tous les graphiques ouverts
                return classes

            else:
                print("Option invalide. Veuillez réessayer.")

        # Passer à n suivant uniquement après avoir classé avec l'option 1 ou 2
        plt.close()
        if action in {"1", "2"}:
            print(classes)
            current_n += 1
            
    return classes

# Demander à l'utilisateur de spécifier N
try:

    N = int(input("Entrez la valeur de N (maximum de n) : "))
    S=user_interface(N)
    print(S)
except ValueError:
    print("Pas un N valide")


""" 2 à 101

{ 

1 : L'élément neutre

--------nombres premiers------------
    
'nombres premiers ': [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101],

---------nombres composés = (2 pics) ----------

'nombres purs': [4, 8, 9, 16, 25, 27, 32, 49, 64, 81],
'plateau': [6, 15, 26, 30, 35, 36, 77, 84], 
'2 pics': [10, 14, 20, 21, 22, 28, 33, 34, 38, 39, 40, 44, 46, 50, 51, 52, 55, 56, 57, 58, 62, 63, 65, 68, 69, 74, 76, 80, 82, 85, 86, 87, 88, 91, 92, 93, 94, 95, 98, 99, 100], 
'deviation': [12, 18, 24, 45, 48, 54, 60, 72, 75, 96],
'plateau et pic ': [42, 66, 70, 78], 
'double deviation': [90]}


"""