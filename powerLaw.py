import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import powerlaw
import sys


def powerLawCalculations(filename):
    # Read the file:
    stanford_graph = pd.read_csv(filename, delimiter=',')
    stanford_graph.set_index(stanford_graph.columns[0], inplace=True)

    # # Plot degree distribution:
    # For counts
    plt.figure()
    plt.bar(stanford_graph['degree'], stanford_graph['count'])
    plt.xlabel('Degree')
    plt.ylabel('Count')
    plt.title('Degree-count distribtution')


#     #For log of counts
#     plt.bar(stanford_graph['degree'], np.log(stanford_graph['count']))
#     plt.xlabel('Degree')
#     plt.ylabel('Count')
#     plt.title('Amazon degreel log(count) distribtution')

    results = powerlaw.Fit(stanford_graph['count'])
    print(results.power_law.alpha)

    # Checking if the data fits power law distribution or exponential distribution
    R, p = results.distribution_compare(
        "power_law", "exponential", normalized_ratio=True)
    print("*" * 50)
    print("Checking wrt exponential distribution")
    if R < 0:
        print("R value of ", R,
              "indicates that the distribution is not power-law in nature")
    else:
        print("R value of ", R,
              "indicates that the distribution is power-law in nature")
    if p > 0.05:
        print("p value for this comparison is ", p,
              "which indicates that neither distribution is a significantly strong fit")
    else:
        print("p value for this comparison is ", p,
              "which indicates that power-law is a significantly strong fit")

    print("*" * 50)
    # Checking if the data fits power law distribution or log normal distribution
    print("*" * 50)
    print("Checking wrt log normal distribution")
    R, p = results.distribution_compare(
        "power_law", "lognormal", normalized_ratio=True)
    if R < 0:
        print("R value of ", R,
              "indicates that the distribution is not power-law in nature")
    else:
        print("R value of ", R,
              "indicates that the distribution is power-law in nature")
    if p > 0.05:
        print("p value for this comparison is ", p,
              "which indicates that neither distribution is a significantly strong fit")
    else:
        print("p value for this comparison is ", p,
              "which indicates that power-law is a significantly strong fit")

    print("*" * 50)
    plt.figure()
    fig_ccdf = results.plot_ccdf()
    results.power_law.plot_ccdf(ax=fig_ccdf, color="r", linestyle="--")

    plt.figure()
    plt.scatter(np.log(stanford_graph['degree']),
                np.log(stanford_graph['count']))
    plt.xlabel('Degree')
    plt.ylabel('Count')
    plt.title('Log-log distribtution')


powerLawCalculations(sys.argv[1])
