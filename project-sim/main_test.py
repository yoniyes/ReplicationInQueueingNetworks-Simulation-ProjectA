import numpy as np
import NewFunction

# set initial value w(T)
# randomWorkLoads = [0,1,2,3,4,5,6,7,8,9,10]
randomWorkLoads = range(5, -5, -1)

# T = 10

# wAVG is 1st moment vector
# secondMoment is obviouly 2nd moment vector
wAvg = [0]*len(randomWorkLoads)
SecondMoment = [0]*len(randomWorkLoads)

# i=1

for i in range(1, len(randomWorkLoads)):
    wAvg[i] = NewFunction.CalcAvgWorkLoad(i, wAvg[i-1], np.sum(randomWorkLoads[0:i]))
    SecondMoment[i] = NewFunction.CalcSecondMoment(SecondMoment[i-1], wAvg[i - 1], 0.5) # have to put the right ratio
    print "i is " + str(i)
    print "total workload: " + str(np.sum(randomWorkLoads[0:i]))
    print "wAvg is " + str(wAvg[i])
    print "SecondMoment is " + str(SecondMoment[i])

stdv = NewFunction.CalcSTDV(wAvg)

print "STDV is " + str(stdv)

windowAVG = NewFunction.CalcWindowAvg(wAvg)

print "Avg of window is " + str(windowAVG)

NewFunction.ConvergenceCheck(stdv, windowAVG,0.5 )