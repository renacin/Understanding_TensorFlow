# Name:                                             Renacin Matadeen
# Date:                                                01/20/2018
# Title                                          Understanding Seaborn
#
#
# ----------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
from matplotlib import pyplot
import pandas as pd
import seaborn as sns

#plt.style.use('seaborn-darkgrid')

# ----------------------------------------------------------------------------------------------------------------------
"""
    Understanding Seaborn:
        + Seaborn works ontop of Matplotlib
            - Makes some things easier

        + Histograms In Seaborn
            plt.hist(data["Mpg"], alpha=.3)
            sns.rugplot(data["Mpg"])
            plt.show()

        + Scatter Matrix
            pd.scatter_matrix(data, alpha=0.2) # Works Quite Well In Pandas

# """
# ----------------------------------------------------------------------------------------------------------------------

# Import Car_Data
raw_data_df = pd.read_csv("C:/Users/renac/Documents/Programming/Python/Tensorflow/Understanding_TensorFlow/Data/Cars_Data.csv")
data = raw_data_df.copy()

# Remove "?" from dataset
data = data[(data != '?').all(axis=1)]

# Explore Data With Scatter Matrix
# pyplot.figure(figsize=(15, 15))
# sns.distplot(data["Mpg"])
sns.pairplot(data.astype(float), plot_kws={"s": 5})
plt.show()


# ----------------------------------------------------------------------------------------------------------------------
