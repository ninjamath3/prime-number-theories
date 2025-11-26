def coef(m,p):
    a=get_near_power(m,p)
    if m>1:
        if p**a==m:
            return a
        else:
            return coef(m-p**a,p)
    else:
        return 0

    
def get_near_power(m,p):
    a=0
    while p**a<=m:
        a+=1
    return a-1

L=[]
for i in range(2,1000,1):
    s=0
    for j in range(2,i,1):
        s+=coef(i,j)
        if s!=0:
            break
    if s==0:
        L.append(i)
    else:
        s=0

print(L)