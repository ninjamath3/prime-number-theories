from matplotlib import pyplot as plt

def trouver_val(f,e,d,c,b,a,graine=3):
    u0=1
    u1=graine
    u2=graine**2
    u3=graine**3
    u4=graine**4
    u5=graine**5
    return a*u0+b*u1+c*u2+d*u3+e*u4+f*u5
        
def get_L_u(m,g):
    dico_u={}
    i=0
    while m>((g**i)-1)/2:
        dico_u["u{}".format(i)]=g**i
        i+=1
    L=list(dico_u.items())
    return L        
        
def iterate_aui(ui,min_a,max_a,L_aui):     
    if min_a==max_a:
        return L_aui
    else :
        L_aui.append(ui*min_a)
        return iterate_aui(ui,min_a+1,max_a,L_aui)
        
def iterate_ui(L_u,min_a,max_a):
    dico_L_aui={}
    for i in range(0,len(L_u),1):
        dico_L_aui["u{}".format(i)]=iterate_aui(L_u[i][1],min_a,max_a,[])
    return dico_L_aui    

def get_iterate_list(L):
    H=[]
    for i in range(len(L[0][1])):
        H.append(i)
    return H
    
def magic_funtion(L,iter_L):
    R=[]
    for val in iter_L:
        for i in range(len(L[0][1])):
            R.append((i,val))
    return R

def get_list_index(magic_list):
    magic_string=str(magic_list)
    magic_string_1=magic_string.replace("(","")
    magic_string_2=magic_string_1.replace(")","")
    L_index=eval(magic_string_2)
    return L_index

def get_tuple(L_index,iter,L_prod,g,signe_m):
    iter=iter-1
    combi=[]
    result=[]
    for j in range(len(L_prod)):
        combi.append(L_index[iter-j])
        result.append(combi[j]-int(((g-1)/2)))
        if signe_m:
            result[j]=-result[j]
    
    return result
    
def get_abcd(m,g):
    signe_moins=False
    if m==0:
        return [0]
    if m==1:
        return[1]
    if m==-1:
        return[-1]
    if m<0:
        signe_moins=True
        m=-m
    min_a=-int((g-1)/2)
    max_a=int(((g-1)/2))+1 
    L=get_L_u(m,g)
    dico_prod_a_ui=iterate_ui(L,min_a,max_a)
    L_prod=list(dico_prod_a_ui.items())
    s=0
    R=magic_funtion(L_prod,get_iterate_list(L_prod))
    for i in range(len(L_prod)-2):
        R=magic_funtion(L_prod,R)    
    list_index=get_list_index(R)
    iter=0
    for calcul in range(len(L_prod[0][1])**len(L_prod)):
        for i in range(0,len(L_prod),1):
            s+=L_prod[i][1][list_index[iter]]
            iter+=1

        if s==m:
            result=get_tuple(list_index,iter,L_prod,g,signe_moins)
            return result
        else:
            s=0

def zero_padding(L,n):
    if len(L)==n:
        return L
    else:
        L.append(0)
        L= [L[-1]] + L[:-1]
        return zero_padding(L,n)

def somme_abcd(A,B):
    if len(A)!=len(B):
        if len(A)>len(B):
            B=zero_padding(B,len(A))
        else:
            A=zero_padding(A,len(B))
    R=[]
    for i in range(len(A)):
        R.append(A[i]+B[i])
    return R
            
                       

                     
P=[3,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,257,263,269,271,277,281,313,317,331,337,347]

for i in range(6,20,1):
    m=i

    A=get_abcd(m,3)
    print(A)


    C=get_abcd(m,7)
    print(C)


    R=somme_abcd(A,C)

    print("\n avec m={} resultat : {}".format(m,R))




