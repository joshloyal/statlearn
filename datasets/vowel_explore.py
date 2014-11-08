from pandas import read_csv
from urllib import urlopen
import seaborn as sns
import matplotlib.pyplot as plt

url = "http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/vowel.train"
page = urlopen(url)
df = read_csv(url)

#sns.lmplot('x.1', 'x.2', df, hue='y', fit_reg=False)
#plt.show()
