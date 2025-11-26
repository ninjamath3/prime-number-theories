import sympy as sp
from matplotlib import pyplot as plt 

qn, dn, c, a, b, i, j, n, alpha, q0, d0, nmod = sp.symbols('qn dn c a b i j n alpha q0 d0 nmod')

def apply_delta_condition(delta_num):
    if delta_num==0:
        q0=1
        d0=3
        nmod=0
        c=3

    if delta_num==1:
        q0=5
        d0=7
        nmod=1
        c=7

    if delta_num==2:
        q0=1
        d0=9
        nmod=2
        c=1

    if delta_num==3:
        q0=10
        d0=11
        nmod=3
        c=9
        
    alpha=(n-nmod)/4
    dn=10*alpha+d0
    qn=c*alpha+q0
    
    return q0,d0,nmod,c,alpha,dn,qn

N_P=[]
X=[]
Y=[]
for delta in [0]:
    q0,d0,nmod,c,alpha,dn,qn=apply_delta_condition(1)

    for a in range(0,20,1):
        for b in range(0,10,1):
            for j in range(0,20,1):
                eq= sp.Eq((a+b*qn)/dn, j)
                sol=sp.solve(eq,n)
                if sol!=[] and sol[0].is_integer:
                    X.append(j)
                    Y.append(sol[0])
                    
plt.plot(X,Y)
plt.show()
                