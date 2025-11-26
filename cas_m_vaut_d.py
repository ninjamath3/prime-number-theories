import sympy as sp

# Définir les symboles nécessaires
alpha_nM,q_nD,d0M,j,alpha_nD,d0D,kmij,i,c,n4M,n4D,nM,nD,q0D=sp.symbols('alpha_nM q_nD d0M j alpha_nD d0D kmij i c n4M n4D nM nD q0D')

def resoudre_cas(delta_m,delta_d):
    n4D=delta_d
    n4M=delta_m
    
    if delta_d==0:
        c=3
        q0D=1
        d0D=3
        
    if delta_d==1:
        c=7
        q0D=5
        d0D=7
        
    if delta_d==2:
        c=1
        q0D=1
        d0D=9
        
    if delta_d==3:
        c=9
        q0D=10
        d0D=11
        
    if delta_m==0:
        d0M=3
        
    if delta_m==1:
        d0M=7
        
    if delta_m==2:
        d0M=9
        
    if delta_m==3:
        d0M=11

    alpha_nD=(nD-n4D)/4
    alpha_nM=(nM-n4M)/4
    q_nD=c*alpha_nD+q0D
    kmij=10*j+(1-i)*c
    
    eq1=sp.Eq(10*alpha_nM+d0M,kmij*10*alpha_nD+d0D*kmij)
    eq2=sp.Eq(alpha_nM+q_nD*d0M,10*j*alpha_nD+j*d0D)
    eq3=sp.Eq(alpha_nM,j*(10*alpha_nD+d0D)+(1-i)*q_nD)
    eq4=sp.Eq(d0M,i-1)
        
    system=[eq1,eq2]
    solutions=sp.solve(system,(i,j))
    sp.pprint(solutions)
        
def cas_general():
    alpha_nD=(nD-n4D)/4
    alpha_nM=(nM-n4M)/4
    q_nD=c*alpha_nD+q0D
    kmij=10*j+(1-i)*c
    
    eq1=sp.Eq(10*alpha_nM+d0M,kmij*10*alpha_nD+d0D*kmij)
    eq2=sp.Eq(alpha_nM+q_nD*d0M,10*j*alpha_nD+j*d0D)
    eq4=sp.Eq(alpha_nM,j*(10*alpha_nD+d0D)+(1-i)*q_nD)
    eq3=sp.Eq(d0M,i-1)
        
    system=[eq1,eq2]
    solutions=sp.solve(system,(i,j))
    sp.pprint(solutions)
    
cas_general()
resoudre_cas(1,0)