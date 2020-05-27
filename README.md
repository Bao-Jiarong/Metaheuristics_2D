## Metaheuristics in Python
The implemented metaheuristics are:

* Random Search (RS),
* Simple Descent (SD),
* Deepest Descent (DS),
* Multistart Descent (MD),
* Tabu Search (TS),
* Simulated Annealing (SA),
* Threshold Accept (TA).

All of the implemented algorithms can be used to find the minimum of 2D function.  
For example : f(x) = (x-2)^2.

### Requirement
```
python==3.7.0
numpy==1.18.1
```
### How to use

Open test.py you will find some examples
```
import src.mt as meta

def h(x):
    return (x-4)**2-4

mt = meta.MT(f=h,x0=10,T=100,low=-10,high=10)

x = mt.random_search()
print("x =",x,"h(x) =",h(x))
```
