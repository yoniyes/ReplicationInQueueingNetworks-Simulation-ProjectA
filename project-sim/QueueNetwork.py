from Queue import Queue
import numpy as np


class QueueNetwork:
    """Class for operating a queue network in discrete time."""

    ##
    # Initialize a queue network given a size, service rates and initial workloads.
    # If only a size was given, all queues will have service rate = 1, workload = 0.
    # All parameters are integers where size > 0, all service rates > 0 and workloads >= 0.
    ##
    def __init__(self, size, serviceRates=[], workloads=[]):
        _size = int(size)
        _serviceRates = serviceRates
        _workloads = workloads
        if _size < 1:
            raise Exception("Illegal size for QueueNetwork")
        for i in range(len(_serviceRates)):
            _serviceRates[i] = int(_serviceRates[i])
            if _serviceRates[i] < 1:
                raise Exception("Illegal service rates for QueueNetwork")
        for i in range(len(_workloads)):
            _workloads[i] = int(_workloads[i])
            if _workloads[i] < 0:
                raise Exception("Illegal workloads for QueueNetwork")
        if not _serviceRates:
            _serviceRates = [1 for i in range(_size)]
        if not _workloads:
            _workloads = [0 for i in range(_size)]
        self.queues = [Queue(_serviceRates[i], _workloads[i]) for i in range(_size)]
        self.size = _size

    ##
    # Resets all queues to service rate of 1 and workload of 0.
    ##
    def reset(self):
        [self.queues[i].reset() for i in range(self.size)]

    ##
    # Returns a list of the current service rates.
    ##
    def getServiceRates(self):
        serviceRates = []
        [serviceRates.append(self.queues[i].getServiceRate()) for i in range(self.size)]
        return serviceRates

    ##
    # Returns a list of the current workloads.
    ##
    def getWorkloads(self):
        workloads = []
        [workloads.append(self.queues[i].getWorkload()) for i in range(self.size)]
        return workloads

    ##
    # Returns the total workload in the network.
    ##
    def getTotalWorkload(self):
        return np.sum([q.getWorkload() for q in self.queues])

    ##
    # Returns the number of queues in the network.
    ##
    def getSize(self):
        return self.size

    ##
    # Sets the service rates to the given sizes. service rates are integers and are > 0.
    ##
    def setServiceRates(self, serviceRates = []):
        if not serviceRates or len(serviceRates) != self.size:
            raise Exception("Illegal number of service rates for setServiceRates")
        for i in range(self.size):
            if int(serviceRates[i]) < 1:
                raise Exception("Illegal service rates for setServiceRates")
        [self.queues[i].setServiceRate(int(serviceRates[i])) for i in range(self.size)]

    ##
    # Sets the workloads to the given sizes. workloads are integers and are >= 0.
    ##
    # def setWorkloads(self, workloads = []):
    #     if not workloads or len(workloads) != self.size:
    #         raise Exception("Illegal number of workloads for setWorkloads")
    #     for i in range(self.size):
    #         if int(workloads[i]) < 0:
    #             raise Exception("Illegal workloads for setWorkloads")
    #     [self.queues[i].setWorkload(int(workloads[i])) for i in range(self.size)]

    def setWorkloads(self, queues = [], newWorkload = 0, policy = "BATCHING"):
        if queues == [] or newWorkload < 0:
            return
        if policy == "BATCHING":
            for q in queues:
                self.queues[q].setWorkload(newWorkload)
        elif policy == "RANDOM":
            for q in queues:
                self.queues[q].setWorkload(np.max([newWorkload, self.queues[q].getWorkload()]))

    ##
    # Adds the given workload to the specified queues.
    # Notice: this method can be used to decrease the workload if passing a negative value.
    ##
    def addWorkload(self, workload, index = []):
        _workload = int(workload)
        _index = []
        for i in range(len(index)):
            if int(index[i]) < 0 or int(index[i]) > self.size - 1:
                raise Exception("Illegal index for addWorkload")
            _index.append(int(index[i]))
        [self.queues[_index[i]].addWorkload(_workload) for i in range(len(_index))]

    ##
    # Reduces the amount of workload in every queue by its service rate and increments the time that passed.
    ##
    def advanceTimeSlot(self):
        [self.queues[i].advanceTimeSlot() for i in range(self.size)]