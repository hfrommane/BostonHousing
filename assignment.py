# 载入此项目所需要的库
import numpy as np
import pandas as pd
import visuals as vs  # Supplementary code
from sklearn.model_selection import ShuffleSplit

# 载入波士顿房屋的数据集
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis=1)

# 完成
print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))
