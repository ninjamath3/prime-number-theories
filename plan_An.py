import Libnbq as Lbq #type:ignore


p=2.8
d0=1
Ap=Lbq.get_A_p(p,d0,10)
Lbq.plot_m(Ap)
print("d={}".format(10*p+d0))

print("\n q={}".format(Ap[1][2]-Ap[0][2]))
print("\n q={}".format(9*p+1))
