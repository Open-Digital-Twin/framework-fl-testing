import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt



true = np.random.randint(0, 10, size=100)
pred = np.random.randint(0, 10, size=100)
labels = np.arange(10)
target_names = list("ABCDEFGHI")

clf_report = classification_report(true,
                                pred,
                                labels=labels,
                                target_names=target_names,
                                output_dict=True)
sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
plt.savefig('save_as_a_png.png')