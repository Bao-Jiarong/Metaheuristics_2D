'''
 *  Author : Bao Jiarong
 *  Contact: bao.salirong@gmail.com
 *
 *  Created  On: 2020-05-27
 *  Modified On: 2020-05-27
 '''
from .algo import *

class MT(Algo):
    #----------------------------------------------------------
    # Constructor
    #----------------------------------------------------------
    def __init__(self,f,x0,T,low,high,eps=1e-3,verbose=False):
        Algo.__init__(self,T,f,eps,verbose)
        self.x0   = x0
        self.low  = low
        self.high = high

    #----------------------------------------------------------
    # Random_Search
    # Info (En): https://en.wikipedia.org/wiki/Random_search
    #----------------------------------------------------------
    def random_search(self):
        self.err = []
        x    = self.x0
        fmin = self.f(x)
        xmin = x

        for i in range(self.T):
            # Algorithm
            x  = randfloat(self.low, self.high)
            fx = self.f(x)
            if fx < fmin:
                fmin = fx
                xmin = x

            # Loss
            self.err.append(fmin)

            # Progress
            self.display(i,xmin)

            # Early stopping
            if abs(fmin) < self.eps:
                break

        return xmin

    #----------------------------------------------------------
    # Simple_Descent
    # Info (En):
    #----------------------------------------------------------
    def simple_descent(self):
        self.err = []
        x    = self.x0
        xmin = x
        fmin = self.f(x)

        for i in range(self.T):
            # Algorithm
            x  = randfloat(x - self.high, x + self.high)
            fx = self.f(x)
            if fx < fmin:
                fmin = fx
                xmin = x

            # Loss
            self.err.append(fmin)

            # Progress
            self.display(i,xmin)

            # Early stopping
            if abs(fmin) < self.eps:
                break

        return xmin

    #----------------------------------------------------------
    # Deepest_Descent
    # Info (En):
    #----------------------------------------------------------
    def deepest_descent(self):
        self.err = []
        x    = self.x0
        xmin = x
        fmin = self.f(x)
        m    = 10

        for j in range(self.T):
            # Algorithm
            s = []
            t = []
            for i in range(m):
                r = randfloat(xmin - self.high, xmin + self.high)
                s.append(r)
                t.append(self.f(r))
            index = np.array(t).argmin()
            x  = s[index]
            fx = t[index]
            if fx < fmin:
                xmin = x
                fmin = fx

            # Loss
            self.err.append(fmin)

            # Progress
            self.display(j,xmin)

            # Early stopping
            if abs(fmin) < self.eps:
                break

        return xmin

    #----------------------------------------------------------
    # Multistart_Descent
    # Info (En):
    #----------------------------------------------------------
    def multistart_descent(self):
        self.err = []
        fmin = float('inf')
        xmin = float('inf')

        for i in range(self.T):
            self.x0 = randfloat(self.low, self.high)
            x  = self.deepest_descent()
            fx = self.f(x)
            if fx < fmin:
                fmin = fx
                xmin = x

            # Loss
            self.err.append(fmin)

            # Progress
            self.display(i,xmin)

            # Early stopping
            if abs(fmin) < self.eps:
                break

        return xmin

    #----------------------------------------------------------
    # Tabu_Search
    # Info (En): https://en.wikipedia.org/wiki/Tabu_search
    # Info (Ch): https://zh.wikipedia.org/wiki/禁忌搜索
    #----------------------------------------------------------
    def tabu_search(self, tabu_size = 100):
        self.err = []
        xmin = self.x0
        fmin = self.f(self.x0)
        tabu = np.array([])

        for i in range(self.T):
            # Algorithm
            s = []
            t = []
            for _ in range(tabu_size):
                r = randfloat(xmin - self.high, xmin + self.high)
                if r not in tabu:
                    s.append(r)
                    t.append(self.f(r))

            index = np.array(t).argmin()
            x     = s[index]
            fx    = t[index]
            tabu  = np.append(tabu, x)

            if tabu.shape[0] >= tabu_size:
                tabu = np.delete(tabu,0)

            if fx < fmin:
                fmin = fx
                xmin = x

            # Loss
            self.err.append(fmin)

            # Progress
            self.display(i,xmin)

            # Early stopping
            if abs(fmin) < self.eps:
                break

        return xmin

    #----------------------------------------------------------
    def decrease_temperature(self,t,method,alpha,gamma,delta):
        if method == "linear":
            t = alpha * t
        elif method == "discrete":
            t = t - alpha
        elif method == "exponential":
            t = t * np.exp((-delta*t)/gamma)
        return t

    #----------------------------------------------------------
    def metropolis_rule(self,x,x1,T):
        fx = self.f(x)
        fx1= self.f(x1)
        if fx1 <= fx:
            return x1
        else:
            p = np.exp(-(fx1 - fx)/(T + self.eps))
            r = randfloat(0,1)
            if r <= p:
                return x1
        return x

    #----------------------------------------------------------
    # Simulated Annealing
    # Info (En): https://en.wikipedia.org/wiki/Simulated_annealing
    # Info (Ch): https://zh.wikipedia.org/wiki/模拟退火
    #----------------------------------------------------------
    def simulated_annealing(self,method="linear",min_t=4,alpha=0.1,gamma=3,delta=4):
        self.x0 = randfloat(self.low, self.high)
        xmin    = self.x0
        fmin    = self.f(self.x0)
        self.err= []
        t       = self.T
        m       = 10

        while t > min_t:
            for j in range(m):
                x1 = randfloat(xmin - self.high, xmin + self.high)
                x0 = self.metropolis_rule(xmin, x1, t)
                fx0= self.f(x0)
                if fx0 < fmin:
                    fmin = fx0
                    xmin = x0

            # Loss
            self.err.append(fmin)

            # Progress
            self.display(t,xmin)

            # Early stopping
            if abs(fmin) < self.eps:
                break

            t = self.decrease_temperature(t, method, alpha, gamma, delta)

        return xmin

    #----------------------------------------------------------
    # Threshold Accept
    # Info (En):
    # Info (Ch):
    #----------------------------------------------------------
    def threshold_accept(self,method="linear",alpha=0.1,gamma=3,delta=4):
        x     = self.x0
        fx    = self.f(x)
        xmin  = x
        fmin  = fx
        moved = True
        t     = self.T
        m     = 10
        self.err= []

        while moved == True:
            for i in range(m):
                x0    = randfloat(xmin - self.high, xmin + self.high)
                fx0   = self.f(x0)
                delta = fx0 - fx
                moved = True
                if delta < 0 or delta < t:
                    x = x0
                    fx= fx0
                else:
                    moved = False

                if fx < fmin:
                    fmin = fx
                    xmin = x

            # Loss
            self.err.append(fmin)

            # Progress
            i = i + 1
            self.display(i,xmin)

            # Early stopping
            if abs(fmin) < self.eps:
                break

            t = self.decrease_temperature(t, method, alpha, gamma, delta)

        return xmin
