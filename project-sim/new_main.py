import math
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit
import os.path
import re
from QueueNetwork import QueueNetwork

# calculate the next average workload in the avg workload vector
def calcAvgWorkLoad(T, AvgWorkLoad_prev, TotalWorkLoad):
    T = float(T)
    AvgWorkLoad_prev = float(AvgWorkLoad_prev)
    TotalWorkLoad = float(TotalWorkLoad)
    return (1.0 / T) * ((T - 1) * AvgWorkLoad_prev + TotalWorkLoad)

# checks if standard deviation is small enough
def convergenceCheck(stdv, avg, epsilon):
    return True if (float(avg) > 0.00001 and (float(stdv) / float(avg)) < float(epsilon)) else False


n = 6
# Assumption: 2*T_min << T_max
T_min = 50000
T_max = 5000000
probs = [0.9]
alpha = 10
beta = 1000
d_vals = [1, 2, 3]
policies = ["RANDOM", "BATCHING"]
lambda_granularity_batching = 100
random_policy_edge = int(float(lambda_granularity_batching) * 0.1)
workloadLimit = 2000
eps = 0.05
bucketingFactor = 2.0

first_sim = True
randomizedArrivalsFilename = "n=" + str(n) + ",T_max=" + str(T_max) + ",p=" + str(probs) + ",d=" + str(d_vals) + ",a=" \
                             + str(alpha) + ",b=" + str(beta) + ",granularity=" + str(lambda_granularity_batching)\
                             + ",arrivals.txt"
resultsFilename = "n=" + str(n) + ",T_max=" + str(T_max) + ",p=" + str(probs) + ",d=" + str(d_vals) + ",a=" \
                             + str(alpha) + ",b=" + str(beta) + ",granularity=" + str(lambda_granularity_batching)\
                             + ",results.txt"

# seed = 308001
seed = 20389574
rnd = np.random.RandomState(seed=seed)
net = QueueNetwork(n)
histogramOfJobsLifetime = {}
sumOfHistogram = {}
lim_avg_workload = {}
lambda_param = {}
lambda_granularity = {}
for policy in policies:
    lambda_granularity[policy] = lambda_granularity_batching
    if policy == "RANDOM":
        lambda_granularity[policy] += random_policy_edge
    lambda_param[policy] = []
    lim_avg_workload[policy] = []
    sumOfHistogram[policy] = []
    histogramOfJobsLifetime[policy] = []

lambda_param_tmp = []
# Assumption: first policy in simulation is RANDOM.
if first_sim:
    first_sim = False
    for policy in policies:
        # Generate an effective mu for the current probability of alpha and for every redundancy.
        mu_effective = [1.0 / (alpha * (1 - pow((1 - probs[0]), d)) + beta * pow((1 - probs[0]), d)) for d in d_vals]
        # Generate the tested lambdas for every effective mu.
        lambda_param[policy] = [
            [(x / float(lambda_granularity_batching)) * mu for x in range(0, lambda_granularity[policy] + 1)]
            for mu in mu_effective
        ]
        for ii in range(len(d_vals)):
            d = d_vals[ii]
            num_of_batches = int(math.ceil(n / float(d)))
            mu = mu_effective[ii] * num_of_batches
            if policy == "RANDOM":
                lambda_param_tmp = [lambda_param[policy][ii][jj] * num_of_batches
                                    for jj in range(lambda_granularity[policy] + 1)]

    if not os.path.isfile(randomizedArrivalsFilename):
        fh = open(randomizedArrivalsFilename, 'w')
        for l_p in lambda_param_tmp:
            if l_p == 0:
                continue
            arrivals = rnd.choice(range(1, T_max + 1), int(l_p * T_max), replace=False)
            arrivals.sort()
            prev_arrival = arrivals[0]
            arrivals_string = ""
            for idx in range(len(arrivals) - 1):
                tmp = arrivals[idx + 1]
                arrivals[idx + 1] -= prev_arrival
                arrivals_string += str(arrivals[idx+1]) + " "
                prev_arrival = tmp
            fh.write(arrivals_string + "\n")
        fh.close()

for prob in probs:
    if not os.path.isfile(resultsFilename):
        ts = time.time()
        print "p = " + str(prob)
        for policy in policies:
            # Initialize a matrix where each line will contain the avg workload in different lambdas for a specific redundancy.
            lim_avg_workload[policy] = [
                [
                    0 for i in range(lambda_granularity[policy] + 1)
                ] for x in range(len(d_vals))
            ]
            histogramOfJobsLifetime[policy] = [
                [
                    0 for i in range(2*T_max + 1)
                ] for x in range(len(d_vals))
            ]

            for ii in range(len(d_vals)):
                d = d_vals[ii]
                num_of_batches = int(math.ceil(n / float(d)))
                mu = mu_effective[ii] * num_of_batches
                if d == 1:
                    lambda_param_tmp = [lambda_param[policy][ii][jj] * num_of_batches for jj in
                                        range(lambda_granularity["BATCHING"] + 1)]
                else:
                    lambda_param_tmp = [lambda_param[policy][ii][jj] * num_of_batches for jj in
                                        range(lambda_granularity[policy] + 1)]
                iteration = 0
                print "\t\td = " + str(d_vals[ii])
                for l_p in lambda_param_tmp:
                    iteration += 1
                    windowSize = np.max([int((float(l_p) / mu) * 2.0 * T_min), T_min])
                    print "\t\t\tWindow size = " + str(windowSize)
                    print "\t\t\trunning sim for (lambda / mu) = [" + str(float(l_p) / mu) + "] " + \
                          "(iteration " + str(iteration) + "/" + str(lambda_granularity[policy] + 1) + ")"
                    windowVar = 0.0
                    windowAvg = 0.0
                    wAvg = [0.0]
                    t = 0

                    # Run simulation for the current system settings.
                    if l_p > 0:
                        l_p_div = int(l_p)
                        l_p_rem = l_p-l_p_div
                        # Randomize arrivals only for non-deterministic arrival process.
                        arrivals = []
                        if l_p_rem > 0:
                            with open(randomizedArrivalsFilename, 'r') as fh:
                                for i, line in enumerate(fh):
                                    if i == iteration - 2:
                                        arrivals = line.split(" ")
                                        arrivals.pop()
                                        arrivals = [int(x) for x in arrivals]

                        # Advance time up until prior to the randomized arrival.
                        for nextArrival in arrivals:
                            for i in range(nextArrival - 1):
                                # Every time-slot, add the definite arrivals.
                                for definiteArrival in range(l_p_div):
                                    # Add workload.
                                    if policy == "BATCHING":
                                        # FIXME: wrong assumption that n % d = 0
                                        # assumption: the init state is (0,0,0,...,0)
                                        addedWorkload = np.min(rnd.choice([alpha, beta], d, p=[prob, 1.0 - prob]))
                                        chosenBatch = rnd.randint(num_of_batches)
                                        queuesChosen = filter(lambda x: x < n,
                                                              range(chosenBatch * d, (chosenBatch + 1) * d))
                                        if iteration == lambda_granularity_batching - 1 and t > T_min:
                                            histogramOfJobsLifetime[policy][ii][int(int(net.getWorkloads()[queuesChosen[0]] + addedWorkload) / (bucketingFactor*alpha))] += 1
                                        net.addWorkload(addedWorkload, index=queuesChosen)
                                    elif policy == "RANDOM":
                                        randomWorkload = rnd.choice([alpha, beta], d, p=[prob, 1.0 - prob])
                                        queuesChosen = rnd.choice(n, size=d, replace=False)
                                        workloadInQueuesChosen = [net.getWorkloads()[i] for i in queuesChosen]
                                        workloadInQueuesChosen = [workloadInQueuesChosen[i] + randomWorkload[i] for i in range(d)]
                                        minInQueuesChosenPlusArrival = np.min(workloadInQueuesChosen)
                                        if iteration == lambda_granularity_batching - 1 and t > T_min:
                                            histogramOfJobsLifetime[policy][ii][int(minInQueuesChosenPlusArrival / (bucketingFactor*alpha))] += 1
                                        net.setWorkloads(queues=queuesChosen, newWorkload=minInQueuesChosenPlusArrival,
                                                         policy="RANDOM")
                                net.advanceTimeSlot()
                                t += 1
                                wAvg.append(calcAvgWorkLoad(t, wAvg[t - 1], net.getTotalWorkload()))
                                if t % windowSize == 0:
                                    windowAvg = np.max([np.average(wAvg[t - windowSize:]), 1])
                                    windowVar = np.var(wAvg[(t - windowSize):])
                                # Check for convergence.
                                if iteration == lambda_granularity[policy] + 1:
                                    if t > 2 * T_max or (
                                            windowAvg > lim_avg_workload[policy][ii][iteration - 2] and windowAvg >= workloadLimit):
                                        break
                                elif ((t > T_min and convergenceCheck(math.sqrt(windowVar), windowAvg, eps))
                                      or t > T_max):
                                    if iteration - 1 == 0 or t > T_max or windowAvg > lim_avg_workload[policy][ii][iteration - 2]:
                                        break
                            # Every time-slot, add the definite arrivals.
                            for definiteArrival in range(l_p_div):
                                # Add workload.
                                if policy == "BATCHING":
                                    # FIXME: wrong assumption that n % d = 0
                                    # assumption: the init state is (0,0,0,...,0)
                                    addedWorkload = np.min(rnd.choice([alpha, beta], d, p=[prob, 1.0 - prob]))
                                    chosenBatch = rnd.randint(num_of_batches)
                                    queuesChosen = filter(lambda x: x < n, range(chosenBatch * d, (chosenBatch + 1) * d))
                                    if iteration == lambda_granularity_batching - 1 and t > T_min:
                                        histogramOfJobsLifetime[policy][ii][int(int(net.getWorkloads()[queuesChosen[0]] + addedWorkload) / (bucketingFactor*alpha))] += 1
                                    net.addWorkload(addedWorkload, index=queuesChosen)
                                elif policy == "RANDOM":
                                    randomWorkload = rnd.choice([alpha, beta], d, p=[prob, 1.0 - prob])
                                    queuesChosen = rnd.choice(n, size=d, replace=False)
                                    workloadInQueuesChosen = [net.getWorkloads()[i] for i in queuesChosen]
                                    workloadInQueuesChosen = [workloadInQueuesChosen[i] + randomWorkload[i] for i in range(d)]
                                    minInQueuesChosenPlusArrival = np.min(workloadInQueuesChosen)
                                    if iteration == lambda_granularity_batching - 1 and t > T_min:
                                        histogramOfJobsLifetime[policy][ii][int(minInQueuesChosenPlusArrival / (bucketingFactor*alpha))] += 1
                                    net.setWorkloads(queues=queuesChosen, newWorkload=minInQueuesChosenPlusArrival,
                                                     policy="RANDOM")
                            # Add workload.
                            if policy == "BATCHING":
                                # FIXME: wrong assumption that n % d = 0
                                # assumption: the init state is (0,0,0,...,0)
                                addedWorkload = np.min(rnd.choice([alpha, beta], d, p=[prob, 1.0 - prob]))
                                chosenBatch = rnd.randint(num_of_batches)
                                queuesChosen = filter(lambda x: x < n, range(chosenBatch * d, (chosenBatch + 1) * d))
                                if iteration == lambda_granularity_batching - 1 and t > T_min:
                                    histogramOfJobsLifetime[policy][ii][int(int(net.getWorkloads()[queuesChosen[0]] + addedWorkload) / (bucketingFactor*alpha))] += 1
                                net.addWorkload(addedWorkload, index=queuesChosen)
                            elif policy == "RANDOM":
                                randomWorkload = rnd.choice([alpha, beta], d, p=[prob, 1.0 - prob])
                                queuesChosen = rnd.choice(n, size=d, replace=False)
                                workloadInQueuesChosen = [net.getWorkloads()[i] for i in queuesChosen]
                                workloadInQueuesChosen = [workloadInQueuesChosen[i] + randomWorkload[i] for i in range(d)]
                                minInQueuesChosenPlusArrival = np.min(workloadInQueuesChosen)
                                if iteration == lambda_granularity_batching - 1 and t > T_min:
                                    histogramOfJobsLifetime[policy][ii][int(minInQueuesChosenPlusArrival / (bucketingFactor*alpha))] += 1
                                net.setWorkloads(queues=queuesChosen, newWorkload=minInQueuesChosenPlusArrival,
                                                 policy="RANDOM")
                            net.advanceTimeSlot()
                            t += 1
                            wAvg.append(calcAvgWorkLoad(t, wAvg[t - 1], net.getTotalWorkload()))
                            if t >= windowSize:
                                if t % windowSize == 0:
                                    windowAvg = np.max([np.average(wAvg[t - windowSize:]), 1])
                                    windowVar = np.var(wAvg[(t - windowSize):])
                            # Check for convergence.
                            if iteration == lambda_granularity[policy] + 1:
                                if t > 2 * T_max or (
                                        windowAvg > lim_avg_workload[policy][ii][iteration - 2] and windowAvg >= workloadLimit):
                                    break
                            elif ((t > T_min and convergenceCheck(math.sqrt(windowVar), windowAvg, eps))
                                  or t > T_max):
                                if iteration - 1 == 0 or t > T_max or windowAvg > lim_avg_workload[policy][ii][iteration - 2]:
                                    break

                    print "\t\t\tStopped at t = " + str(t) + "\n"
                    lim_avg_workload[policy][ii][iteration - 1] = windowAvg
                    net.reset()

        ts = time.time() - ts
        print "Simulation for p = " + str(prob) + " took [ " + str(float(ts)/60.0) + " ] minutes.\n"

        # Save results to file.
        fh = open(resultsFilename, 'w')
        print "Writing results to [ " + resultsFilename + " ]\n"
        for policy in policies:
            fh.write("#" + policy + " WORKLOAD\n")
            for ii in range(len(d_vals)):
                for avgWorkload in lim_avg_workload[policy][ii]:
                    fh.write(str(avgWorkload) + " ")
                fh.write("\n")
            fh.write("#" + policy + " TAIL\n")
            for ii in range(len(d_vals)):
                sumOfHistogramAux = [0, histogramOfJobsLifetime[policy][ii][0]]
                for k in range(1, len(histogramOfJobsLifetime[policy][ii])):
                    sumOfHistogramAux.append(sumOfHistogramAux[k] + histogramOfJobsLifetime[policy][ii][k])
                sumOfHistogramAux = [float(sumOfHistogramAux[k]) / float(sumOfHistogramAux[-1]) for k in range(int(len(sumOfHistogramAux)))]
                sumOfHistogramAux = sumOfHistogramAux[:np.argmax(sumOfHistogramAux) * 2]
                for sumSoFar in sumOfHistogramAux:
                    fh.write(str(sumSoFar) + " ")
                fh.write("\n")

    # Read from results file.
    print "Reading results from [ " + resultsFilename + " ]\n"
    with open(resultsFilename, 'r') as fh:
        policy = ""
        dataType = {}
        for line in fh:
            line = line.rstrip()
            if line == "":
                continue
            # matchObj = ""
            matchObj = re.match(r'^#(\w+?)\s+(\w+)', line)
            if matchObj is not None:
                policy = matchObj.group(1)
                dataType = matchObj.group(2)
                continue
            if dataType == "WORKLOAD":
                lim_avg_workload[policy].append([float(x) for x in line.split(" ")])
            elif dataType == "TAIL":
                sumOfHistogram[policy].append([float(x) for x in line.split(" ")])


    def curve_fit_func(t, a, b):
        return a * np.exp(b * t)

    def curve_fit_func2(x, a, b, c, d):
        return a*x**8 + b*x**4 + c*x**2 + d*x


    print "Plotting for all d"
    plotted_d_1 = False
    for policy in policies:
        for ii in range(len(d_vals)):
            d = d_vals[ii]
            if plotted_d_1 and d == 1:
                continue
            plotted_d_1 = True
            num_of_batches = int(math.ceil(n / float(d)))
            mu = mu_effective[ii] * num_of_batches
            lambda_param_tmp = [lambda_param[policy][ii][jj] * num_of_batches for jj in range(lambda_granularity[policy] + 1)]
            lambda_param_fitted = [lp / (float(n) * mu_effective[0]) for lp in lambda_param_tmp]
            lambda_param_fitted = np.array(lambda_param_fitted, dtype=float)
            lim_avg_workload[policy][ii] = np.array(lim_avg_workload[policy][ii], dtype=float)
            # try:
                # popt, pcov = curve_fit(curve_fit_func, lambda_param_fitted, lim_avg_workload[policy][ii], maxfev=10000)
            # except RuntimeError:
            #     # next
            #     continue
            # plt.plot(lambda_param_fitted, curve_fit_func(lambda_param_fitted, *popt), label="Fitted Curve d=" + str(d_vals[ii]) + ', Policy = ' + str(policy))
            if d == 1:
                plt.plot(lambda_param_fitted, lim_avg_workload[policy][ii], 'o',
                         label="Average workload samples d=" + str(d_vals[ii]))
            else:
                plt.plot(lambda_param_fitted, lim_avg_workload[policy][ii], 'o', label="Average workload samples d=" + str(d_vals[ii]) + ', Policy = ' + str(policy))
            plt.title(r'$n$ = ' + str(n) + ', $p$ = ' + str(prob) + ', $\\alpha$ = ' + str(alpha) +
                      ', $\\beta$ = ' + str(beta))
            plt.axvline(x=lambda_param_fitted[lambda_granularity_batching], linestyle='dashed')
            plt.xlabel(r'$\frac{\lambda}{\mu_{d=1}}$')
            plt.ylabel(r'Average workload $(\lim W)$')
            plt.legend(loc='upper right')
    plt.show()

    for ii in range(len(d_vals)):
        print "Plotting for d=" + str(d_vals[ii])
        for policy in policies:
            d = d_vals[ii]
            if policy == 'RANDOM' and d == 1:
                continue
            num_of_batches = int(math.ceil(n / float(d)))
            mu = mu_effective[ii] * num_of_batches
            lambda_param_tmp = [lambda_param[policy][ii][jj] * num_of_batches for jj in range(lambda_granularity[policy] + 1)]
            lambda_param_fitted = [lp / (float(n) * mu_effective[0]) for lp in lambda_param_tmp]
            lambda_param_fitted = np.array(lambda_param_fitted, dtype=float)
            lim_avg_workload[policy][ii] = np.array(lim_avg_workload[policy][ii], dtype=float)
            try:
                popt, pcov = curve_fit(curve_fit_func, lambda_param_fitted, lim_avg_workload[policy][ii], maxfev=10000)
            except RuntimeError:
                # next
                continue
            plt.plot(lambda_param_fitted, curve_fit_func(lambda_param_fitted, *popt), label="Fitted Curve, Policy = " + str(policy))
            plt.plot(lambda_param_fitted, lim_avg_workload[policy][ii], 'o', label="Original average workload samples, Policy = " + str(policy))
            plt.title(r'$n$ = ' + str(n) + ', $p$ = ' + str(prob) + ', $\\alpha$ = ' + str(alpha) +
                      ', $\\beta$ = ' + str(beta) + ', $d$ = ' + str(d_vals[ii]))
            plt.axvline(x=lambda_param_fitted[lambda_granularity_batching], linestyle='dashed')
            plt.xlabel(r'$\frac{\lambda}{\mu_{d=1}}$')
            plt.ylabel(r'Average workload $(\lim W)$')
            plt.legend(loc='upper left')
        plt.show()
    plt.show()

    last_point = len(sumOfHistogram['RANDOM'][0]) - 1
    for policy in policies:
        for ii in range(len(d_vals)):
            extension = [1 for i in range(last_point - len(sumOfHistogram[policy][ii]))]
            sumOfHistogram[policy][ii] += extension
            plt.plot(range(int(len(sumOfHistogram[policy][ii]))), sumOfHistogram[policy][ii], label="d = " + str(d_vals[ii]) + ", Policy =  " + str(policy))
            plt.title(r'$n$ = ' + str(n) + ', $p$ = ' + str(prob) + ', $\\alpha$ = ' + str(alpha) +
                      ', $\\beta$ = ' + str(beta))
            plt.xlabel(r'$k$ (Time in system)')
            plt.ylabel(r'Ratio of tasks done in up to $k$ time to all tasks in $' + str(2*T_max) + '$')
            plt.legend(loc='lower right')
    plt.axhline(y=0.9, linestyle='dashed')
    plt.show()
