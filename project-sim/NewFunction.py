import math
import numpy as np


# calc the next avgerage work load in the avgworkload vector


def CalcAvgWorkLoad (T, AvgWorkLoad_prev, TotalWorkLoad):
    # T = float(T)
    # AvgWorkLoad_prev = float(AvgWorkLoad_prev)
    # TotalWorkLoad = float(TotalWorkLoad)
    return (1.0/T)*((T-1)*AvgWorkLoad_prev + TotalWorkLoad)


# calc the average value of the avg vector -  for normalizing the stdv


def CalcWindowAvg (AvgWorkLoadWindow):
    return np.mean(AvgWorkLoadWindow)


# find stdv of avgworkload vector


def CalcSTDV (AvgWorkLoadWindow):
    return np.std(AvgWorkLoadWindow)


# Calculation of 2nd moment to calc stdv by definition. NOTE!!! this calc is only valid when the process is ergodic
# and the process is W(t) = max{0, W(t-1) + Arrival(t)*Workload(t) - 1}.


def CalcSecondMoment (SecondMoment_prev, FirstMoment_prev,lambdaToMiuRatio):
    return SecondMoment_prev+2*(lambdaToMiuRatio-1)*FirstMoment_prev-lambdaToMiuRatio+1


def CalcConstantC (FirstMoment_prev, FirstMoment_current):
    return FirstMoment_prev^2-2*FirstMoment_prev*FirstMoment_current


def CalcSTDV_byDefinition (T, C, SecondMoment_current):
    return math.sqrt((C + SecondMoment_current)*(1.0-(1.0/T))^2)


# checks if STDV is small enough


def ConvergenceCheck (STDV, windowAvg, epsilon):
    if float(STDV)/windowAvg < epsilon:
        print ("converged")
        return True
    return False


# the random variables part


def RandomArrivals (lambda_param):
    return np.random.geometric(lambda_param)



