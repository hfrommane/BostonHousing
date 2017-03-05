# 载入此项目所需要的库
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import visuals as vs  # Supplementary code
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer

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
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=0)

print("Training and testing split was successful.")


# print("Test size: {:,d}".format(len(X_test)))

# 问题 3- 训练及测试
# Produce learning curves for varying training set sizes and maximum depths
# vs.ModelLearning(features, prices)
# 它们是一个决策树模型在不同最大深度下的表现 depth = 1, 3, 6, 10
# 每一条曲线都直观的显示了随着训练数据量的增加，模型学习曲线的训练评分和测试评分的变化。
# 红色：训练曲线的评分
# 绿色：测试曲线的评分

# 问题 4 - 学习数据
# vs.ModelComplexity(X_train, y_train)
# 它展示了一个已经经过训练和验证的决策树模型在不同最大深度条件下的表现
# 我选择 max_depth = 3
# 增加训练点数量对：
# 训练曲线（红线）：对于非常少的训练点，训练曲线为或接近于1.这是由于过度拟合（决策树足够灵活以解释少数训练点）。
#                 然后，当添加更多训练点时，训练曲线下降，因为决策树模型不能再解释所有方差（由于模型偏差或由于固有的随机性）。
#                 它似乎慢慢收敛到0.8左右的值。
# 测试曲线（绿线）：对于非常少的训练点，测试曲线为或接近0.这是由于过度拟合（决策树完全解释了少数训练点，
#                 但这些不是群体的代表）。测试曲线然后快速上升并收敛到略低于训练曲线的值。这表明良好的泛化。
# 我不相信在这种情况下更多的培训点会有所帮助。两条曲线彼此接近，似乎收敛良好。更多的数据只会帮助我们使用一个更复杂的模型。
# 侧注：关于曲线的标准偏差，我认为他们相当小，以信任曲线（阴影区域很小，不重叠），这是我上面的语句的先决条件。

# 问题 5- 偏差与方差之间的权衡取舍
# 在max_depth为1的情况下，模型受到高偏差的影响。与图中的其他结点相比，训练和验证分数都相对较低。
#                        这表明模型不能解释数据中的方差，无论如何调整它。
# 在max_depth为10的情况下，模型受到高方差的影响。训练得分远高于验证得分。特别地，训练得分高于max_depth的每个其他值。
#                        然而，验证分数在大约4到5的max_depth处是最高的，而对于较大的值则是下降。
#                        这表明模型的方差对于这些较大的值变得太大了。

# 问题 6- 最优模型的猜测
# 我选择max_depth = 4

# 问题 7- 网格搜索（Grid Search）
# 一般来说：网格搜索是一种为不能直接优化的模型参数找到好的值的技术。它通过在模型参数上定义网格，
# 然后评估网格上每个点的模型性能（使用验证集（或CV），而不是训练数据）来工作。然后，您可以选择网格上看起来效果最好的点。
# 示例：网格通常用于识别良好的模型复杂性，实际上，我们将其应用于关于复杂度曲线的任务。
# 在这个任务中，我们不知道max_depth的哪个值最适合应用于波士顿房屋数据集的决策树。
# 我们在max_depth值[1,2，...，10]上定义了一个网格，并为从1到10的每个值评估了决策树的性能。
# 这使我们能够估计max_depth和得分之间的关系。
# 我们使用我们对这种关系的知识，通过寻找一个max_depth值来选择一个很好的max_depth值，增加它的值不会显着提高。

# 问题 8- 交叉验证
# k折交叉验证：对于测试，我们可以将数据分成用于训练模型的一个训练集和用于评估模型性能的另一个测试集。
# 例如，这是上面使用的测试程序（train_test_split）。这是浪费，因为我们使用一部分数据仅用于训练，
# 另一部分用于测试。我们可以通过将数据划分为k个折叠，即整个数据集的k个相等大的块来做得更好。
# 然后，我们一个接一个地遍历k个块，并使用当前块进行模型验证，剩余的k-1个块用于训练。
# 我们最终得到的是经过训练和评估的模型，我们使用了整个数据集进行验证。
# 平均验证分数给我们一个单一的验证分数。它比我们只使用一个分裂更可靠。

# 好处是我们可以更可靠地估计网格搜索期间各种参数配置的模型性能（估计中的方差较小）。
# 在网格搜索中，存在过度拟合验证集的危险，因为我们多次使用它来评估网格上的不同点的性能并选择提供良好性能的点。
# 因此，随着越来越多的网格点，我们越来越可能找到一个只是偶然好的点。
# 通过交叉验证，过拟合问题得到缓解，因为我们的有效验证集大小更大。这是以k倍的单次分裂的计算复杂度为代价的。

# 练习：训练模型
def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a
        decision tree regressor trained on the input data [X, y]. """

    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)

    regressor = DecisionTreeRegressor(random_state=0)

    params = {'max_depth': range(1, 11)}

    scoring_fnc = make_scorer(performance_metric)

    grid = GridSearchCV(regressor, params, scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


# 问题 9- 最优模型
# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))
# 一致的

# 问题 10 - 预测销售价格
# Produce a matrix for client data
client_data = [[5, 17, 15],  # Client 1
               [4, 32, 22],  # Client 2
               [8, 3, 12]]  # Client 3

# Show predictions
for i, price in enumerate(reg.predict(client_data)):
    print("Predicted selling price for Client {}'s home: ${:,.2f}".format(i + 1, price))

# 误差曲线
y_prediction = reg.predict(X_test)
vs.ErrorFigure(y_prediction, y_test)

# 灵敏度
# 最优模型不一定是鲁棒模型。有时，模型太复杂或太简单，无法充分推广到新数据。有时，模型可能使用不适合所给数据结构的学习算法。
# 其他时候，数据本身可能太嘈杂或包含太少的样本，以允许模型充分捕获目标变量 - 即模型欠配合。
# 运行下面的代码单元，fit_model使用不同的训练和测试集运行该函数十次，以查看特定客户端的预测如何随着其训练的数据而变化。
vs.PredictTrials(features, prices, fit_model, client_data)

# 问题11 -适用性
# 近40年前收集的数据与今天的价格无关，因为市场发生变化，房价和货币价值等其他因素也将发生变化。
#
# 每个家庭提供的数据不足以在当今市场中做出准确的预测。例如，您可以开始考虑诸如该地区的公园或城市中的大学的功能，以更好地预测房子的价格。
#
# 该模型在特定时间点使用来自某个城市的数据进行训练，因此该模型可能会不及来自不同类型城市（例如农村城市）的变量数据，如问题1的答案中所讨论的。
