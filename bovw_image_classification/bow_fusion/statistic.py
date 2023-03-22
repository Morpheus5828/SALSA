import statistics


def compute_statistics(data):
    mean = statistics.mean(data)
    standard_deviation = statistics.stdev(data, xbar=mean)
    return mean, standard_deviation
