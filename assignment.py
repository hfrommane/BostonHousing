# 载入此项目所需要的库
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import visuals as vs  # Supplementary code
from sklearn.model_selection import ShuffleSplit

# 载入波士顿房屋的数据集
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis=1)

# 完成
# print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))

# 练习：基础统计运算
# 目标：计算价值的最小值
minimum_price = np.min(prices)

# 目标：计算价值的最大值
maximum_price = np.max(prices)

# 目标：计算价值的平均值
mean_price = np.mean(prices)

# 目标：计算价值的中值
median_price = np.median(prices)

# 目标：计算价值的标准差
std_price = np.std(prices)

# Show the calculated statistics
# 目标：输出计算的结果
print("Statistics for Boston housing dataset:\n")
print("Minimum price: ${:,.2f}".format(minimum_price))
print("Maximum price: ${:,.2f}".format(maximum_price))
print("Mean price: ${:,.2f}".format(mean_price))
print("Median price ${:,.2f}".format(median_price))
print("Standard deviation of prices: ${:,.2f}".format(std_price))


# 问题1 - 特征观察
# 'RM' 是该地区中每个房屋的平均房间数量；
# 'LSTAT' 是指该地区有多少百分比的房东属于是低收入阶层（有工作但收入微薄）；
# 'PTRATIO' 是该地区的中学和小学里，学生和老师的数目比（学生/老师）。

# 练习：定义衡量标准
def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between
        true and predicted values based on the metric chosen. """
    score = r2_score(y_true, y_predict)
    return score


# 问题2 - 拟合程度
# Calculate the performance of this model
# score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
# print("Model has a coefficient of determination, R^2, of {:.3f}.".format(score))
# 0.923 r2得分接近1，说明这个模型已成功地描述了目标变量的变化。

# 练习: 数据分割与重排
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=10)

print("Training and testing split was successful.")

print("Test size: {:,d}".format(len(X_test)))


# 问题 3- 训练及测试
