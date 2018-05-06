from collections import Counter
import math
import numpy as np
import matplotlib.pyplot as plt
from QueueNetwork import QueueNetwork


################################################################
# Used to generate random scenarios for task-replication simulation.
# @p, @alpha, @beta define a random variable:
# workload = { @alpha w.p. @prob; @beta w.p. (1 - @prob) }.
# @lambda_param defines another random variable:
# arrival = { True w.p. @lambda_param; False w.p. (1 - @lambda_param) }.
# @batch_size and the optional @last_batch size are used in the
# calculation of the minimum workload in each batch.
# The function generates a (@T x @batches) matrix.
# If in time-slot 0<t<=T an arrival[t] = False, the whole line
# of the generated matrix will be '0's - meaning, the batches
# workload will not grow. If arrival[t] = True, the only non-zero
# element in line t of the generated matrix will be:
# 	generated_matrix[t][chosen_batch[t]]
# meaning, an arrival occurred in time-slot t and it was assigned
# to the batch that was randomly chosen in that time-slot.
#
# Raises an exception if parameters are not valid.
################################################################
def randomize_init_matrix(prob, alpha, beta, lambda_param, T, batches, batch_size, last_batch_size=0, debug=False, givenArrivals=[]):
    # check parameter validity.
    # if prob < 0.0 or prob > 1.0:
    #     raise Exception("@prob is not between 0 and 1.")
    # elif math.floor(alpha) < 1 or math.floor(beta) < 1:
    #     raise Exception("@alpha and @beta should be >= 1.")
    # elif math.floor(alpha) > math.floor(beta):
    #     raise Exception("@alpha should be <= @beta.")
    # elif lambda_param < 0.0 or lambda_param > 1.0:
    #     raise Exception("@lambda_param is not between 0 and 1.")
    # elif math.floor(T) < 1:
    #     raise Exception("@T is not >= 1.")
    # elif batches < 1:
    #     raise Exception("@batches is not >= 1.")
    # elif batch_size < 1:
    #     raise Exception("@batch_size is not >= 1.")
    # elif last_batch_size < 0 or last_batch_size > batch_size:
    #     raise Exception("@last_batch_size is not between 0 and @batch_size.")

    residue_exists = 0
    if last_batch_size > 0:
        residue_exists = 1
    queue_num = batches * batch_size - residue_exists * (batch_size - last_batch_size)

    # Randomize arrivals.
    arrivals = givenArrivals
    if arrivals == []:
        arrivals = np.random.choice([False, True], int(T), p=[1.0 - lambda_param, lambda_param])
        # arrivals = np.random.RandomState(100).choice([False, True], int(T), p=[1.0 - lambda_param, lambda_param])

    # Randomize batch choosing.
    chosen_batch = []
    if last_batch_size == 0:
        chosen_batch = np.random.choice(batches, int(T))
        # chosen_batch = np.random.RandomState(100).choice(batches, int(T))
    else:
        p_batches = [float(batch_size) / queue_num for i in range(batches - 1)]
        p_batches.append(float(last_batch_size) / queue_num)
        eps = 0.000001
        assert (1.0 + eps > np.sum(p_batches) > 1.0 - eps), \
            "ERROR!\np_batches = " + str(p_batches) + "\np_batches.sum = " + str(np.sum(p_batches))
        chosen_batch = np.random.choice(batches, int(T), p=p_batches)
        # chosen_batch = np.random.RandomState(100).choice(batches, int(T), p=p_batches)

    # Randomize workloads.
    # Initialize a @T x @batches matrix.
    workloads = [[0 for x in range(batches)] for x in range(int(T))]
    # For each time-slot, for each batch, determine the workload.
    for t in range(int(T)):
        for batch in range(batches):
            power = batch_size
            if batch == batches - 1 and last_batch_size > 0:
                power = last_batch_size
            q = pow(1.0 - prob, power)
            workloads[t][batch] = np.random.choice([int(alpha), int(beta)], 1, p=[1.0 - q, q])[0]
            # workloads[t][batch] = np.random.RandomState(100).choice([int(alpha), int(beta)], 1, p=[1.0 - q, q])[0]

    # Calculate for every time-slot if there's an arrival to a batch.
    effective_workloads = [[0 for x in range(batches + 1)] for x in range(int(T))]
    for t in range(int(T)):
        if (arrivals[t]):
            effective_workloads[t][chosen_batch[t]] = workloads[t][chosen_batch[t]]
            effective_workloads[t][batches] = 1

    if debug:
        arrival_num = 0
        for arrival in arrivals:
            if arrival:
                arrival_num += 1
        summed_lines = np.sum(effective_workloads, axis=1)
        d = Counter(summed_lines)
        assert (d[0] == T - d[int(alpha)] - d[int(beta)]), \
            "##################vec_err##################"
        print "### randomize_init_matrix ###"
        print "Queues: " + str(queue_num)
        print "Redundancy: " + str(batch_size)
        print "Time-slots: " + str(int(T))
        print "Arrivals: " + str(arrival_num)
        print "Alpha's: " + str(d[int(alpha)])
        print "Beta's: " + str(d[int(beta)])

    return effective_workloads, arrivals


n = 10
T = 15000
probs = [0.95]#, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
alpha = 1
beta = 100
d_vals = [i + 1 for i in range(n)]
lambda_granularity = 100

net = QueueNetwork(n)
for prob in probs:
    #mu = 1.0 / (alpha * prob + beta * (1 - prob))
	mu_effective = [1.0 / (alpha * (1-(1-prob)**d) + beta * (1 - prob)**d) for d in d_vals]
	#lambda_param = [[(1-(1/float(x)) + 0.005) * mu for mu in mu_effective] for x in range(1, lambda_granularity + 1)]
	lambda_param = [[(1-(1/float(x))) * mu for mu in mu_effective] for x in range(1, lambda_granularity + 1)]
	print "p = " + str(prob)
	lim_avg_workload = [ [] for x in range(len(d_vals)) ]
	for l_p in lambda_param:
		arrivals = []
		for ii in range(len(d_vals)):
			print "d = " + str(d_vals[ii])
			num_of_batches = int(math.ceil(net.getSize()/float(d_vals[ii])))
			workloads, arrivals = randomize_init_matrix(prob, alpha, beta, l_p[ii], T, num_of_batches, d_vals[ii], givenArrivals=arrivals)
			print "\trunning sim for lambda = [" + str(l_p[ii]) + "]"
			workload_timeslot = [0]
			for t in range(T):
				# Only execute if an arrival occurred in timeslot t.
				if (workloads[t][num_of_batches] == 1):
					for q in range(int(math.ceil(n/d_vals[ii]))):
						if workloads[t][q] > 0:
							net.addWorkload(workloads[t][q], [q])
				net.advanceTimeSlot()
				workload_timeslot.append(net.getTotalWorkload())
			tmp = [float(np.sum(workload_timeslot[0:t])) / t for t in range(1, T+1)]
			avg_workload_timeslot = [0]
			avg_workload_timeslot.extend(tmp)
			lim_avg_workload[ii].append(np.average(avg_workload_timeslot[int(T/2):]))
			net.reset()

            # plt.plot(range(T+1), avg_workload_timeslot)
            # plt.xlabel(r'Time')
            # plt.ylabel(r'Average workload')
            # plt.show()

	for ii in range(len(d_vals)):
		l_p = [lambda_param[x][ii] for x in range(len(lambda_param))]
		plt.plot(l_p, lim_avg_workload[ii])
		plt.title(r'$n$ = ' + str(n) + ', $p$ = ' + str(prob) + ', $T$ = ' + str(T) + ', $\\alpha$ = ' + str(alpha) + \
                  ', $\\beta$ = ' + str(beta)) #+ ', $\mu$ = ' + str(int(mu*1000) / 1000.0) + ', $d$ = ' + str(d_vals[ii]))
		plt.xlabel(r'$\lambda$')
		plt.ylabel(r'Average workload $(\lim W)$')
		#plt.savefig('plots/n=' + str(n) + ',p=' + str(prob) + ',T=' + str(T) + ',a=' + str(alpha) + ',b=' + str(beta) + \
        #            ',d=' + str(d_vals[ii]) + '.png')
		#plt.savefig('n=' + str(n) + ',p=' + str(prob) + ',T=' + str(T) + ',a=' + str(alpha) + ',b=' + str(beta) + \
        #            ',d=' + str(d_vals[ii]) + '.png')
	plt.show()

