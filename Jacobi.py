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
ErrorTable2 = [1]
final_time2=[0]
omega= 1.5
eps = 0.001
k = 1
start_time2=time.time()
def GS(A, b, x2):    
  A = np.array(A)
  n = len(A)
  x2new = np.zeros(n)
  
  for k in range(n):
    x2new[k] = (b[k] - sum([A[k, l]*x2[l] 
    for l in range(k+1, n)]) - sum([A[k, l]*x2new[l]
    for l in range(k)]))/A[k, k]
  return x2new

while k <= m and abs(ErrorTable2[k - 1]) > eps: 
      xj = GS(A, b, x2[k - 1])
      x2.append(xj) 
      ej = np.linalg.norm(x2[k] - x2[k - 1])/np.linalg.norm(x2[k]) 
      ErrorTable2.append(ej)
      k+=1
      end_time2=time.time()
      execute_time2=end_time2 - start_time2
      final_time2.append(execute_time2)
      
end_time2=time.time()
execute_time2=end_time2 - start_time2
g=np.array(x2)
s=[x for x in range(100000)]
v=[x for x in s[1:m+1]]

it=pd.DataFrame(v, columns=["Iteration"],dtype=float)
di=[f'x{i}' for i in range(1,n+1)]
dx=pd.DataFrame(g, columns=[di],dtype=float)
dk=[f'x{k}' for k in range(1,n+1)]
dxx=pd.DataFrame(ErrorTable2,columns=['Jacobi Error'])
dxx1=pd.DataFrame(final_time2
                  ,columns=['Jacobi Timing'])
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
print('\n\nJacobi Iterative Time execution:',"%.50f"%execute_time2)
ax=plt.gca()
df1=[it,dxx]
ta=pd.concat(df1,axis=1,join="inner")
ta.plot(kind='line',x='Iteration',y='Jacobi Error',ax=ax)
plt.xlabel("Iterations (k)")
plt.ylabel("Approximation Error")
plt.xlim(0,15)
plt.ylim(0,1)
plt.show()
