'''
 *  Author : Bao Jiarong
 *  Contact: bao.salirong@gmail.com
 *
 *  Created  On: 2020-05-27
 *  Modified On: 2020-05-27
 '''
import src.mt as meta

#---------------------------na-----------------------------
def h(x):
    return (x-4)**2-4

mt = meta.MT(f=h,x0=10,T=100,low=-10,high=10)

#----------------------------mt------------------------------
# mt.set_verbose(True)
print("Random Search")
x = mt.random_search()
print("x =",x,"h(x) =",h(x))
#--------------------------
print("Simple Descent")
x = mt.simple_descent()
print("x =",x,"h(x) =",h(x))
#--------------------------
print("Deepest Descent")
x = mt.deepest_descent()
print("x =",x,"h(x) =",h(x))
#--------------------------
print("Multistart Descent")
x = mt.multistart_descent()
print("x =",x,"h(x) =",h(x))
#--------------------------
print("Tabu Search")
x = mt.tabu_search(tabu_size = 10)
print("x =",x,"h(x) =",h(x))
#--------------------------
print("Simulated Annealing")
x = mt.simulated_annealing(method= "linear", # others: "exponential","discrete"
                           min_t = 4,        # minimum temperature
                           alpha = 0.1,      # between 0~1
                           gamma = 3,
                           delta = 4)
print("x =",x,"h(x) =",h(x))
#--------------------------
print("Threshold Accept")
x = mt.threshold_accept(method = "linear",   # others: "exponential","discrete"
                        alpha  = 0.1,        # between 0~1
                        gamma  = 3,
                        delta  = 4)
print("x =",x,"h(x) =",h(x))
