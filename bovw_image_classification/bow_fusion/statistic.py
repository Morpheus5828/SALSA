import statistics

'''
Computes mean and standard-deviation for a list of float.

input = a list of float
output = mean and standard-deviation of the input list
-- use the library statistics
'''
def compute_statistics(data):
    mean = statistics.mean(data)
    standard_deviation = statistics.stdev(data, xbar=mean)
    return mean, standard_deviation
