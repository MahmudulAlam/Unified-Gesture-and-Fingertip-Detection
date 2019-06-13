import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

conf_mat = np.load('data/conf_mat.npy')

class_name = ['SingleOne', 'SingleTwo', 'SingleThree', 'SingleFour', 'SingleFive',
              'SingleSix', 'SingleSeven', 'SingleEight']

df_cm = pd.DataFrame(conf_mat, class_name, class_name)

sn.set(font_scale=1.3)
sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g', square=True)

plt.xticks(rotation=45)
plt.savefig("data/conf_mat.eps", bbox_inches="tight", pad_inches=0)
plt.show()
