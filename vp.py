import sympy as sp

# Définir les symboles nécessaires
qn, dn, c, a, b, i, j, n, alpha, q0, d0, nmod = sp.symbols('qn dn c a b i j n alpha q0 d0 nmod')

#on se place dans le cas de delta0:
d0=11
q0=10
nmod=3
c=9
gamma_a=2
gamma_b=0


alpha = (n - nmod) / 4
qn = c * alpha + q0
dn = 10 * alpha + d0

# Définir la matrice A
A = sp.Matrix([
    [-10, -1],
    [1, qn]
])

# Définir le vecteur B (termes constants)
B = sp.Matrix([c * dn - 10 * a - b + 10 * j * dn - c * i * dn, a + qn * b - j * dn])

# Vérifier si A est singulière
det_A = A.det()
if det_A == 0:
    print("La matrice est singulière (déterminant nul).")
    # Résoudre le système en utilisant la méthode des solutions paramétriques
    solutions = sp.linsolve((A, B))
    print("Solutions possibles (paramétriques) :")
    factorized_solutions = []
    for sol in solutions:
        # Collecter et factoriser chaque solution
        simplified_sol = [sp.simplify(sp.collect(sp.together(s), [n, nmod, q0])) for s in sol]
        factorized_solutions.append(simplified_sol)
    sp.pprint(factorized_solutions)
else:
    print("La matrice est non singulière (déterminant non nul).")
    # Résoudre directement
    X = A.inv() * B
    print("Solution unique (forme factorisée) :")
    sp.pprint(X)
    X_factorized = [sp.simplify(sp.collect(sp.together(x), [n, nmod, q0])) for x in X]

     # Extraire et afficher uniquement les numérateurs
    numerateurs = [sp.numer(x) for x in X_factorized]  # Extraction des numérateurs
    print("Numérateurs: \n")
    
    sp.pprint(numerateurs)  # Affichage des numérateurs
    
equations=[]
for numerateur in numerateurs:
    # Créer une équation symbolique N = 0
    equation = sp.Eq(numerateur, gamma_a)
    equations.append(equation)
    print("\nequation :\n")
    sp.pprint(equation)
    gamma_a=gamma_b

solutions=sp.solve(equations, (i, n))
# Afficher les solutions pour i et j
print("\nSolutions pour i et n :")
sp.pprint(solutions)

sol=[]
for s in solutions:
    for h in s:
        sol.append(h)

eq2 = sp.Eq(sol[1], n)
print("\n")
sp.pprint(eq2)

j_sol=sp.solve(eq2, j)

print("\n voici j(n) : \n")
sp.pprint(j_sol)
