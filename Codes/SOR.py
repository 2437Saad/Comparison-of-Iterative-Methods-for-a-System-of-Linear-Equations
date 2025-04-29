import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


n= int(input("Enter the order of matrix: "))
elements1=list(map(int,input("Enter the elements of matrix: ").split()))
A=np.array(elements1).reshape(n,n)
elements2=list(map(int,input("Enter the elements of coefficient matrix: ").split()))
b=np.array(elements2).reshape(n)
m=int(input("Number of maximum iteration: "))

x1 = [[0]*n]
x2 = [[0]*n]
x3 = [[0]*n]
ErrorTable3 = [1]
final_time3=[0]
omega= 1.5
eps = 0.001
p = 1
start_time3=time.time()
def SOR(A, b, x3,w):
  A = np.array(A)
  n = len(A)
  x3new = np.zeros(n)
  for p in range(n):
    x3new[p] = (1-w)*x3[p]+w*(b[p] - sum([A[p, q]*x3[q] 
    for q in range(p+1, n)]) - sum([A[p, q]*x3new[q]
    for q in range(p)]))/A[p, p]
  return x3new

while p <= m and abs(ErrorTable3[p - 1]) > eps:
      xk = SOR(A, b, x3[p - 1],omega)
      x3.append(xk) 
      ek = np.linalg.norm(x3[p] - x3[p - 1])/np.linalg.norm(x3[p])  
      ErrorTable3.append(ek)
      p+=1
      end_time3=time.time()
      execute_time3=end_time3 - start_time3
      final_time3.append(execute_time3)    
      
h=np.array(x3)
s=[x for x in range(100000)]
v=[x for x in s[1:m+1]]

it=pd.DataFrame(v, columns=["Iteration"],dtype=float)
di=[f'x{i}' for i in range(1,n+1)]
dx=pd.DataFrame(h, columns=[di],dtype=float)
dk=[f'x{k}' for k in range(1,n+1)]
dxx=pd.DataFrame(ErrorTable3,columns=['Gauss-Seidel Error'])
dxx1=pd.DataFrame(final_time3
                  ,columns=['Gauss-Seidel Timing'])
dff1=[it,dx]
t1=pd.concat(dff1,axis=1,join="inner")
tx1=[t1,dxx1]
gs=pd.concat(tx1,axis=1,join="inner")
print("\nJacobi Iteration:\n\n",gs)
df1=[it,dxx]
ta=pd.concat(df1,axis=1,join="inner")

print('\nError Analysis:\n\n',ta)
df4=[it,dxx1]
taa=pd.concat(df4,axis=1,join="inner")

print('\nTime Analysis:\n\n',taa)
print('\n\nJacobi Iterative Time execution:',"%.50f"%execute_time3)
ax=plt.gca()

dzz=pd.DataFrame(ErrorTable3,columns=['SOR Error'])
df3=[it,dzz]
tc=pd.concat(df3,axis=1,join="inner")
tc.plot(kind='line',x='Iteration',y='SOR Error',ax=ax)
plt.xlabel("Iterations (k)")
plt.ylabel("Approximation Error")
plt.xlim(0,15)
plt.ylim(0,1)
plt.show()



