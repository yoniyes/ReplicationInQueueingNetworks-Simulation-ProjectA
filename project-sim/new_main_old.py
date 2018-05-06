import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from QueueNetwork import QueueNetwork


# TODO: Separate simulation from plotting.
# TODO: Multi-threaded simulation (a thread per redundancy level?).
# TODO: Verify what is curve_fit's approximation function.

# calc the next avg workload in the avg workload vector
def calcAvgWorkLoad(T, AvgWorkLoad_prev, TotalWorkLoad):
    T = float(T)
    AvgWorkLoad_prev = float(AvgWorkLoad_prev)
    TotalWorkLoad = float(TotalWorkLoad)
    return (1.0 / T) * ((T - 1) * AvgWorkLoad_prev + TotalWorkLoad)


# calc the variance of the avg vector.
# FIXME: calculating the variance in the window is not working well for now. Probably won't use this function.
def calcWindowVar(avg, prevWindowVar, AvgWorkloadThatLeftWindow, latestAvgWorkloadInWindow, windowSize):
    return np.max([prevWindowVar - (pow(AvgWorkloadThatLeftWindow - avg, 2) -
                                    pow(latestAvgWorkloadInWindow - avg, 2)) / float(windowSize), 0])


def calcWindowVar2(window, AVG):
    total = 0.0
    for avg in window:
        total += (avg - AVG)**2
    return total / float(len(window))
    # return np.var(window)


# calc the average value of the avg vector.
def calcWindowAvg(prevWindowAvg, AvgWorkloadThatLeftWindow, latestAvgWorkloadInWindow, windowSize):
    return prevWindowAvg - (AvgWorkloadThatLeftWindow - latestAvgWorkloadInWindow) / float(windowSize)


# checks if stdv is small enough
def convergenceCheck(stdv, avg, epsilon):
    # print "ro = " + str(float(stdv) / float(avg)) + ", eps = " + str(float(epsilon)) + ", " + str((float(stdv) / float(avg)) < float(epsilon))
    return True if (float(avg) > 0.00001 and (float(stdv) / float(avg)) < float(epsilon)) else False


n = 10
# Assumption: 2*T_min << T_max
T_min = 2500
T_max = 100000
probs = [0.75]
alpha = 1
beta = 20
d_vals = [1, 2, 5, 10]
lambda_granularity = 100
policy = "BATCHING"
workloadLimit = 2000
eps = 0.05

seed = 203895743
rnd = np.random.RandomState(seed=seed)
net = QueueNetwork(n)
for prob in probs:
    print "p = " + str(prob)
    # Generate an effective mu for the current probability of alpha and for every redundancy.
    mu_effective = [1.0 / (alpha * (1 - pow((1 - prob), d)) + beta * pow((1 - prob), d)) for d in d_vals]
    # Generate the tested lambdas for every effective mu.
    lambda_param = [
        [(x / float(lambda_granularity)) * mu for x in range(0, lambda_granularity + 1)]
        for mu in mu_effective
    ]
    # Initialize a matrix where each line will contain the avg workload in different lambdas for a specific redundancy.
    lim_avg_workload = [[0 for i in range(lambda_granularity + 1)] for x in range(len(d_vals))]

    for ii in range(len(d_vals)):
        # mu = mu_effective[ii]
        d = d_vals[ii]
        num_of_batches = 0
        if policy == "BATCHING":
            num_of_batches = int(math.ceil(n / float(d)))
        elif policy == "RANDOM":
            num_of_batches = n
        mu = mu_effective[ii] * num_of_batches
        lambda_param_tmp = [lambda_param[ii][jj] * num_of_batches for jj in range(lambda_granularity + 1)]
        iteration = 0
        print "\t\td = " + str(d_vals[ii])
        for l_p in lambda_param_tmp: # lambda_param[ii]:
            iteration += 1
            windowSize = np.max([int((float(l_p) / mu) * 2.0 * T_min), T_min])
            print "\t\t\tWindow size = " + str(windowSize)
            print "\t\t\trunning sim for (lambda / mu) = [" + str(float(l_p) / mu) + "] " + \
                  "(iteration " + str(iteration) + "/" + str(lambda_granularity + 1) + ")"
            windowVar = 0.0
            windowAvg = 0.0
            wAvg = [0.0]
            t = 0
            # Run simulation for the current system settings.
            # while l_p > 0:
            #     # Randomize next arrival.
            #     nextArrival = int(rnd.geometric(l_p))
            #     # Advance time up until prior to the randomized arrival.
            #     for i in range(nextArrival - 1):
            #         net.advanceTimeSlot()
            #         t += 1
            #         wAvg.append(calcAvgWorkLoad(t, wAvg[t - 1], net.getTotalWorkload()))
            #         if t % windowSize == 0:
            #             windowAvg = np.max([np.average(wAvg[t-windowSize:]), 1])
            #             # windowVar = calcWindowVar(windowAvg, windowVar, wAvg[t-windowSize], wAvg[t], windowSize)
            #             windowVar = np.var(wAvg[(t-windowSize):])
            #             # windowVar = calcWindowVar2(wAvg[t - windowSize:], windowAvg)
            #         # Check for convergence.
            #         if iteration == lambda_granularity + 1:
            #             if t > 2*T_max or (
            #                     windowAvg > lim_avg_workload[ii][iteration - 2] and windowAvg >= workloadLimit):
            #                 break
            #         elif ((t > T_min and convergenceCheck(math.sqrt(windowVar), windowAvg, eps))
            #               or t > T_max):
            #             if iteration - 1 == 0 or t > T_max or windowAvg > lim_avg_workload[ii][iteration - 2]:
            #                 break
            #
            #     # Add workload.
            #     if policy == "BATCHING":
            #         addedWorkload = np.min(rnd.choice([alpha, beta], d, p=[prob, 1.0 - prob]))
            #         chosenBatch = rnd.randint(num_of_batches)
            #         queuesChosen = range(chosenBatch * d, (chosenBatch + 1) * d)
            #         net.addWorkload(addedWorkload, index=queuesChosen)
            #     elif policy == "RANDOM":
            #         randomWorkload = rnd.choice([alpha, beta], d, p=[prob, 1.0 - prob])
            #         queuesChosen = rnd.choice(n, size=d, replace=False)
            #         addedWorkload = [0 for i in range(n)]
            #         i = 0
            #         for q in queuesChosen:
            #             addedWorkload[q] = randomWorkload[i]
            #             i += 1
            #         net.addWorkloadRandom(addedWorkload)
            #     net.advanceTimeSlot()
            #     t += 1
            #     wAvg.append(calcAvgWorkLoad(t, wAvg[t - 1], net.getTotalWorkload()))
            #     if t >= windowSize:
            #         # windowAvg = np.average(wAvg[1:]) if t == windowSize else \
            #         #     calcWindowAvg(windowAvg, wAvg[t - windowSize], wAvg[t], windowSize)
            #         if t % windowSize == 0:
            #             windowAvg = np.max([np.average(wAvg[t - windowSize:]), 1])
            #             # windowVar = calcWindowVar(windowAvg, windowVar, wAvg[t - windowSize], wAvg[t], windowSize)
            #             windowVar = np.var(wAvg[(t - windowSize):])
            #             # windowVar = calcWindowVar2(wAvg[t - windowSize:], windowAvg)
            #     # Check for convergence.
            #     if iteration == lambda_granularity + 1:
            #         if t > 2*T_max or (windowAvg > lim_avg_workload[ii][iteration - 2] and windowAvg >= workloadLimit):
            #             break
            #     elif ((t > T_min and convergenceCheck(math.sqrt(windowVar), windowAvg, eps))
            #             or t > T_max):
            #         # break
            #         # if iteration - 1 > 0:
            #         #     print "windowAvg = " + str(windowAvg) + ", prev_windowAvg = " + str(lim_avg_workload[ii][iteration - 2])
            #         if iteration - 1 == 0 or t > T_max or windowAvg > lim_avg_workload[ii][iteration - 2]:
            #             break

            # Run simulation for the current system settings.
            if l_p > 0:
                print "lambda = " + str(l_p)
                l_p_div = int(l_p)
                l_p_rem = l_p-l_p_div
                print "div = " + str(l_p_div) + ", rem = " + str(l_p_rem)
                # Randomize arrivals only for non-deterministic arrival process.
                arrivals = []
                if l_p_rem > 0.000001:
                    arrivals = rnd.choice(range(1, T_max + 1), int(l_p_rem * T_max), replace=False)
                    arrivals.sort()
                    prev_arrival = arrivals[0]
                    for idx in range(len(arrivals) - 1):
                        tmp = arrivals[idx + 1]
                        arrivals[idx + 1] -= prev_arrival
                        prev_arrival = tmp
                print "Arrivals for lambda_rem: " + str(len(arrivals))
                # Advance time up until prior to the randomized arrival.
                if l_p_rem <= 0.000001 and l_p_div > 0:
                    l_p_div -= 1
                    arrivals = range(1, T_max + 1)
                for nextArrival in arrivals:
                    for i in range(nextArrival - 1):
                        # Every time-slot, add the definite arrivals.
                        for definiteArrival in range(l_p_div):
                            # Add workload.
                            if policy == "BATCHING":
                                # FIXME: wrong assumption that n % d = 0
                                addedWorkload = np.min(rnd.choice([alpha, beta], d, p=[prob, 1.0 - prob]))
                                chosenBatch = rnd.randint(num_of_batches)
                                queuesChosen = filter(lambda x: x < n, range(chosenBatch * d, (chosenBatch + 1) * d))
                                net.addWorkload(addedWorkload, index=queuesChosen)
                            elif policy == "RANDOM":
                                randomWorkload = rnd.choice([alpha, beta], d, p=[prob, 1.0 - prob])
                                queuesChosen = rnd.choice(n, size=d, replace=False)
                                addedWorkload = [0 for i in range(n)]
                                i = 0
                                for q in queuesChosen:
                                    addedWorkload[q] = randomWorkload[i]
                                    i += 1
                                net.addWorkloadRandom(addedWorkload)
                        net.advanceTimeSlot()
                        t += 1
                        wAvg.append(calcAvgWorkLoad(t, wAvg[t - 1], net.getTotalWorkload()))
                        if t % windowSize == 0:
                            windowAvg = np.max([np.average(wAvg[t - windowSize:]), 1])
                            # windowVar = calcWindowVar(windowAvg, windowVar, wAvg[t-windowSize], wAvg[t], windowSize)
                            windowVar = np.var(wAvg[(t - windowSize):])
                            # windowVar = calcWindowVar2(wAvg[t - windowSize:], windowAvg)
                        # Check for convergence.
                        if iteration == lambda_granularity + 1:
                            if t > 2 * T_max or (
                                    windowAvg > lim_avg_workload[ii][iteration - 2] and windowAvg >= workloadLimit):
                                break
                        elif ((t > T_min and convergenceCheck(math.sqrt(windowVar), windowAvg, eps))
                              or t > T_max):
                            if iteration - 1 == 0 or t > T_max or windowAvg > lim_avg_workload[ii][iteration - 2]:
                                break
                    # Every time-slot, add the definite arrivals.
                    for definiteArrival in range(l_p_div):
                        # Add workload.
                        if policy == "BATCHING":
                            # FIXME: wrong assumption that n % d = 0
                            addedWorkload = np.min(rnd.choice([alpha, beta], d, p=[prob, 1.0 - prob]))
                            chosenBatch = rnd.randint(num_of_batches)
                            queuesChosen = filter(lambda x: x < n, range(chosenBatch * d, (chosenBatch + 1) * d))
                            net.addWorkload(addedWorkload, index=queuesChosen)
                        elif policy == "RANDOM":
                            randomWorkload = rnd.choice([alpha, beta], d, p=[prob, 1.0 - prob])
                            queuesChosen = rnd.choice(n, size=d, replace=False)
                            addedWorkload = [0 for i in range(n)]
                            i = 0
                            for q in queuesChosen:
                                addedWorkload[q] = randomWorkload[i]
                                i += 1
                            net.addWorkloadRandom(addedWorkload)
                    # Add workload.
                    if policy == "BATCHING":
                        # FIXME: wrong assumption that n % d = 0
                        addedWorkload = np.min(rnd.choice([alpha, beta], d, p=[prob, 1.0 - prob]))
                        chosenBatch = rnd.randint(num_of_batches)
                        queuesChosen = filter(lambda x: x < n, range(chosenBatch * d, (chosenBatch + 1) * d))
                        net.addWorkload(addedWorkload, index=queuesChosen)
                    elif policy == "RANDOM":
                        randomWorkload = rnd.choice([alpha, beta], d, p=[prob, 1.0 - prob])
                        queuesChosen = rnd.choice(n, size=d, replace=False)
                        addedWorkload = [0 for i in range(n)]
                        i = 0
                        for q in queuesChosen:
                            addedWorkload[q] = randomWorkload[i]
                            i += 1
                        net.addWorkloadRandom(addedWorkload)
                    net.advanceTimeSlot()
                    t += 1
                    wAvg.append(calcAvgWorkLoad(t, wAvg[t - 1], net.getTotalWorkload()))
                    if t >= windowSize:
                        # windowAvg = np.average(wAvg[1:]) if t == windowSize else \
                        #     calcWindowAvg(windowAvg, wAvg[t - windowSize], wAvg[t], windowSize)
                        if t % windowSize == 0:
                            windowAvg = np.max([np.average(wAvg[t - windowSize:]), 1])
                            # windowVar = calcWindowVar(windowAvg, windowVar, wAvg[t - windowSize], wAvg[t], windowSize)
                            windowVar = np.var(wAvg[(t - windowSize):])
                            # windowVar = calcWindowVar2(wAvg[t - windowSize:], windowAvg)
                    # Check for convergence.
                    if iteration == lambda_granularity + 1:
                        if t > 2 * T_max or (
                                windowAvg > lim_avg_workload[ii][iteration - 2] and windowAvg >= workloadLimit):
                            break
                    elif ((t > T_min and convergenceCheck(math.sqrt(windowVar), windowAvg, eps))
                          or t > T_max):
                        # break
                        # if iteration - 1 > 0:
                        #     print "windowAvg = " + str(windowAvg) + ", prev_windowAvg = " + str(lim_avg_workload[ii][iteration - 2])
                        if iteration - 1 == 0 or t > T_max or windowAvg > lim_avg_workload[ii][iteration - 2]:
                            break

                # FIXME: What if lambda > 1 ??? That happens when mu_effective > 1 ...
                # arrivals = rnd.choice(range(1, T_max + 1), np.minimum(int(l_p * T_max), T_max), replace=False)
                # arrivals.sort()
                # prev_arrival = arrivals[0]
                # for idx in range(len(arrivals) - 1):
                #     tmp = arrivals[idx+1]
                #     arrivals[idx+1] -= prev_arrival
                #     prev_arrival = tmp

                # # Advance time up until prior to the randomized arrival.
                # for nextArrival in arrivals:
                #     for i in range(nextArrival - 1):
                #         net.advanceTimeSlot()
                #         t += 1
                #         wAvg.append(calcAvgWorkLoad(t, wAvg[t - 1], net.getTotalWorkload()))
                #         if t % windowSize == 0:
                #             windowAvg = np.max([np.average(wAvg[t-windowSize:]), 1])
                #             # windowVar = calcWindowVar(windowAvg, windowVar, wAvg[t-windowSize], wAvg[t], windowSize)
                #             windowVar = np.var(wAvg[(t-windowSize):])
                #             # windowVar = calcWindowVar2(wAvg[t - windowSize:], windowAvg)
                #         # Check for convergence.
                #         if iteration == lambda_granularity + 1:
                #             if t > 2*T_max or (
                #                     windowAvg > lim_avg_workload[ii][iteration - 2] and windowAvg >= workloadLimit):
                #                 break
                #         elif ((t > T_min and convergenceCheck(math.sqrt(windowVar), windowAvg, eps))
                #               or t > T_max):
                #             if iteration - 1 == 0 or t > T_max or windowAvg > lim_avg_workload[ii][iteration - 2]:
                #                 break
                #
                #     # Add workload.
                #     if policy == "BATCHING":
                #         # FIXME: wrong assumption that n % d = 0
                #         addedWorkload = np.min(rnd.choice([alpha, beta], d, p=[prob, 1.0 - prob]))
                #         chosenBatch = rnd.randint(num_of_batches)
                #         queuesChosen = filter(lambda x: x < n, range(chosenBatch * d, (chosenBatch + 1) * d))
                #         net.addWorkload(addedWorkload, index=queuesChosen)
                #     elif policy == "RANDOM":
                #         randomWorkload = rnd.choice([alpha, beta], d, p=[prob, 1.0 - prob])
                #         queuesChosen = rnd.choice(n, size=d, replace=False)
                #         addedWorkload = [0 for i in range(n)]
                #         i = 0
                #         for q in queuesChosen:
                #             addedWorkload[q] = randomWorkload[i]
                #             i += 1
                #         net.addWorkloadRandom(addedWorkload)
                #     net.advanceTimeSlot()
                #     t += 1
                #     wAvg.append(calcAvgWorkLoad(t, wAvg[t - 1], net.getTotalWorkload()))
                #     if t >= windowSize:
                #         # windowAvg = np.average(wAvg[1:]) if t == windowSize else \
                #         #     calcWindowAvg(windowAvg, wAvg[t - windowSize], wAvg[t], windowSize)
                #         if t % windowSize == 0:
                #             windowAvg = np.max([np.average(wAvg[t - windowSize:]), 1])
                #             # windowVar = calcWindowVar(windowAvg, windowVar, wAvg[t - windowSize], wAvg[t], windowSize)
                #             windowVar = np.var(wAvg[(t - windowSize):])
                #             # windowVar = calcWindowVar2(wAvg[t - windowSize:], windowAvg)
                #     # Check for convergence.
                #     if iteration == lambda_granularity + 1:
                #         if t > 2*T_max or (windowAvg > lim_avg_workload[ii][iteration - 2] and windowAvg >= workloadLimit):
                #             break
                #     elif ((t > T_min and convergenceCheck(math.sqrt(windowVar), windowAvg, eps))
                #             or t > T_max):
                #         # break
                #         # if iteration - 1 > 0:
                #         #     print "windowAvg = " + str(windowAvg) + ", prev_windowAvg = " + str(lim_avg_workload[ii][iteration - 2])
                #         if iteration - 1 == 0 or t > T_max or windowAvg > lim_avg_workload[ii][iteration - 2]:
                #             break

            print "\t\t\tStopped at t = " + str(t) + "\n"
            lim_avg_workload[ii][iteration - 1] = windowAvg
            net.reset()

            # plt.plot(range(t+1), wAvg)
            # plt.xlabel(r'Time')
            # plt.ylabel(r'Average workload')
            # plt.show()

    # for ii in range(len(d_vals)):
    #     l_p = [lambda_param[x][ii] for x in range(len(lambda_param))]
    #     plt.plot(l_p, lim_avg_workload[ii])
    #     plt.title(r'$n$ = ' + str(n) + ', $p$ = ' + str(prob) + ', $T$ = ' + str(T) + ', $\\alpha$ = ' + str(alpha) +
    #               ', $\\beta$ = ' + str(
    #         beta))  # + ', $\mu$ = ' + str(int(mu*1000) / 1000.0) + ', $d$ = ' + str(d_vals[ii]))
    #     plt.xlabel(r'$\lambda$')
    #     plt.ylabel(r'Average workload $(\lim W)$')
    #     plt.savefig('plots/n=' + str(n) + ',p=' + str(prob) + ',T=' + str(T) + ',a=' + str(alpha) + ',b=' + str(beta) +
    #                 ',d=' + str(d_vals[ii]) + '.png')
    #     plt.show()
    #

    # plt.plot(lambda_param[ii], lim_avg_workload[ii])
    # plt.title(r'$n$ = ' + str(n) + ', $p$ = ' + str(prob) + ', $\\alpha$ = ' + str(alpha) +
    #           ', $\\beta$ = ' + str(beta) + ', $d$ = ' + str(d))
    # # + ', $\mu$ = ' + str(int(mu*1000) / 1000.0))
    # plt.xlabel(r'$\lambda$')
    # plt.ylabel(r'Average workload $(\lim W)$')
    # plt.show()

    def curve_fit_func(t, a, b):
        return a * np.exp(b * t)

    def curve_fit_func2(x, a, b, c, d):
        return a*x**8 + b*x**4 + c*x**2 + d*x

    for ii in range(len(d_vals)):
        print "Plotting d=" + str(d_vals[ii])

        d = d_vals[ii]
        num_of_batches = 0
        if policy == "BATCHING":
            num_of_batches = int(math.ceil(n / float(d)))
        elif policy == "RANDOM":
            num_of_batches = n
        mu = mu_effective[ii] * num_of_batches
        lambda_param_tmp = [lambda_param[ii][jj] * num_of_batches for jj in range(lambda_granularity + 1)]

        lambda_param_fitted = [lp / (float(n) * mu_effective[0]) for lp in lambda_param_tmp]
        # lambda_param_fitted = [lp / (float(n) * mu_effective[0]) for lp in lambda_param[ii]]
        lambda_param_fitted = np.array(lambda_param_fitted, dtype=float)
        lim_avg_workload[ii] = np.array(lim_avg_workload[ii], dtype=float)
        try:
            popt, pcov = curve_fit(curve_fit_func, lambda_param_fitted, lim_avg_workload[ii], maxfev=10000)
        except RuntimeError:
            next
        plt.plot(lambda_param_fitted, curve_fit_func(lambda_param_fitted, *popt), label="Fitted Curve d=" + str(d_vals[ii]))
        # curve_fit(lambda t, a, b: a * np.exp(b * t), lambda_param[ii], lim_avg_workload[ii])  # , p0=(1, 0.1))
        plt.plot(lambda_param_fitted, lim_avg_workload[ii], 'o', label="Original average workload samples d=" + str(d_vals[ii]))
        plt.title(r'$n$ = ' + str(n) + ', $p$ = ' + str(prob) + ', $\\alpha$ = ' + str(alpha) +
                  ', $\\beta$ = ' + str(beta) + ', $d$ = ' + str(d_vals[ii]))
        # + ', $\mu$ = ' + str(int(mu*1000) / 1000.0) + ', $d$ = ' + str(d_vals[ii]))
        plt.xlabel(r'$\frac{\lambda}{\mu_{d=1}}$')
        plt.ylabel(r'Average workload $(\lim W)$')
        plt.legend(loc='upper left')
    plt.show()

    # for ii in range(len(d_vals)):
    #     print "Plotting d=" + str(d_vals[ii])
    #     lambda_param[ii] = np.array(lambda_param[ii], dtype=float)
    #     lim_avg_workload[ii] = np.array(lim_avg_workload[ii], dtype=float)
    #     popt, pcov = curve_fit(curve_fit_func, lambda_param[ii], lim_avg_workload[ii])
    #     plt.plot(lambda_param[ii], curve_fit_func(lambda_param[ii], *popt), label="Fitted Curve")
    #     # curve_fit(lambda t, a, b: a * np.exp(b * t), lambda_param[ii], lim_avg_workload[ii])  # , p0=(1, 0.1))
    #     # plt.plot(lambda_param[ii], lim_avg_workload[ii], 'ro', label="Original average workload samples")
    #     plt.title(r'$n$ = ' + str(n) + ', $p$ = ' + str(prob) + ', $\\alpha$ = ' + str(alpha) +
    #               ', $\\beta$ = ' + str(beta))
    #     # + ', $\mu$ = ' + str(int(mu*1000) / 1000.0) + ', $d$ = ' + str(d_vals[ii]))
    #     plt.xlabel(r'$\lambda$')
    #     plt.ylabel(r'Average workload $(\lim W)$')
    #     # plt.legend(loc='upper left')
    # plt.show()
