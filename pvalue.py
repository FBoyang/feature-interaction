import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats

false_predictions = 0
percentiles = np.array([])

for simulation in range(1, 201):
    null_distribution = np.array([])
    for file in glob.glob("/Users/nicholasliu/Documents/GitHub/feature-interaction/data/PermutationDistribution/"+str(simulation)+"/*.csv"):
        null_distribution = np.append(null_distribution, np.genfromtxt(file, delimiter=",")[0][1])

    threshold = np.percentile(null_distribution, 95)
    observed_interaction = np.genfromtxt("/Users/nicholasliu/Documents/GitHub/feature-interaction/data/InteractionScore/NN_"+str(simulation)+".csv", delimiter=",")[0][1]
    if observed_interaction > threshold:
        false_predictions += 1

    percentiles = np.append(percentiles, sp.stats.percentileofscore(null_distribution, observed_interaction))

print(percentiles)
print(false_predictions)

plt.hist(percentiles, bins=50)
plt.show()
