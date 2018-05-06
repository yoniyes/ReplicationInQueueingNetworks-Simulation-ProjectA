class Queue:
    """This class describes a queue in a network running in discrete time with the ability to add and
    remove workload."""

    ##
    # Given a service rate and an initial workload, initializes a Queue.
    # Both parameters are integers where serviceRate > 0 and workload >= 0.
    ##
    def __init__(self, serviceRate=1, workload=0):
        if int(serviceRate) <= 0 or int(workload) < 0:
            raise Exception("Illegal arguments for Queue __init__")
        self.serviceRate = int(serviceRate)
        self.workload = int(workload)
        self.timePassed = 0

    ##
    # Resets the queue to service rate of 1 and workload of 0.
    ##
    def reset(self):
        self.serviceRate = 1
        self.workload = 0
        self.timePassed = 0

    ##
    # Returns the current workload.
    ##
    def getWorkload(self):
        return self.workload

    ##
    # Sets the workload to the given size. workload is an integer and workload >= 0.
    ##
    def setWorkload(self, workload):
        if int(workload) < 0:
            raise Exception("Illegal arguments for Queue setWorkload")
        self.workload = int(workload)

    ##
    # Adds the given workload to the queue. Makes sure the total workload is >= 0.
    # Notice: this method can be used to decrease the workload if passing a negative value.
    ##
    def addWorkload(self, workload):
        if self.workload + int(workload) < 0:
            self.workload = 0
        else:
            self.workload += int(workload)

    ##
    # Returns the current service rate.
    ##
    def getServiceRate(self):
        return self.serviceRate

    ##
    # Sets the service rate to the given size. service rate is an integer and it is > 0.
    ##
    def setServiceRate(self, serviceRate):
        if int(serviceRate) <= 0:
            raise Exception("Illegal arguments for Queue setServiceRate")
        self.serviceRate = int(serviceRate)

    ##
    # Reduces the amount of workload in Queue by serviceRate and increments the time that passed.
    ##
    def advanceTimeSlot(self):
        self.addWorkload(-self.serviceRate)
        self.timePassed += 1
