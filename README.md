# BostonHousing

## 练习：基础统计运算
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

程序运行结果：

- Minimum price: $105,000.00
- Maximum price: $1,024,800.00
- Mean price: $454,342.94
- Median price $438,900.00
- Standard deviation of prices: $165,171.13

## 问题1 - 特征观察
- 'RM' 是该地区中每个房屋的平均房间数量；
- 'LSTAT' 是指该地区有多少百分比的房东属于是低收入阶层（有工作但收入微薄）；
- 'PTRATIO' 是该地区的中学和小学里，学生和老师的数目比（学生/老师）。

## 练习：定义衡量标准
R^2的数值范围从0至1，表示目标变量的预测值和实际值之间的相关程度平方的百分比。一个模型的R^2值为0还不如直接用平均值来预测效果好；而一个R^2值为1的模型则可以对目标变量进行完美的预测。从0至1之间的数值，则表示该模型中目标变量中有百分之多少能够用特征来解释。

	def performance_metric(y_true, y_predict):
	    """ Calculates and returns the performance score between
	        true and predicted values based on the metric chosen. """
	    score = r2_score(y_true, y_predict)
	    return score

## 问题2 - 拟合程度
0.923 R^2得分接近1，说明这个模型已成功地描述了目标变量的变化。

## 练习: 数据分割与重排
	X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=0)

	print("Training and testing split was successful.")


## 问题 3- 训练及测试
![](https://raw.githubusercontent.com/hfrommane/BostonHousing/master/figure/figure_1.png)
它们是一个决策树模型在不同最大深度下的表现 depth = 1, 3, 6, 10

每一条曲线都直观的显示了随着训练数据量的增加，模型学习曲线的训练评分和测试评分的变化。

**红色：训练曲线的评分  绿色：测试曲线的评分**


## 问题 4 - 学习数据
它展示了一个已经经过训练和验证的决策树模型在不同最大深度条件下的表现

**我选择 max_depth = 3**

增加训练点数量对曲线的影响：

- 训练曲线（红线）：对于非常少的训练点，训练曲线为或接近于1。这是由于过度拟合（决策树足够灵活以解释少数训练点）。然后，当添加更多训练点时，训练曲线下降，因为决策树模型不能再解释所有方差（由于模型偏差或由于固有的随机性）。它似乎慢慢收敛到0.8左右的值。
- 测试曲线（绿线）：对于非常少的训练点，测试曲线为或接近0。这是由于过度拟合（决策树完全解释了少数训练点，但这些不是群体的代表）。测试曲线然后快速上升并收敛到略低于训练曲线的值。这表明良好的泛化。

我不认为在这种情况下更多的培训点会有所帮助。两条曲线彼此接近，收敛良好。更多的数据只会帮助我们构建一个更复杂的模型。

![](https://raw.githubusercontent.com/hfrommane/BostonHousing/master/figure/figure_2.png)

## 问题 5- 偏差与方差之间的权衡取舍
- 在max_depth为1的情况下，模型受到高偏差的影响。与图中的其他结点相比，训练和验证分数都相对较低。这表明无论如何调整它，模型都不能解释数据中的方差。
- 在max_depth为10的情况下，模型受到高方差的影响。训练得分远高于验证得分。特别地，训练得分高于max_depth的每个其他值。然而，验证分数在大约4到5的max_depth处是最高的，而对于较大的值则是下降。这表明模型的方差对于这些较大的值变得太大了。

## 问题 6- 最优模型的猜测
**我选择max_depth = 4**

## 问题 7- 网格搜索（Grid Search）
一般来说：网格搜索是一种为不能直接优化的模型参数找到好的值的技术。它通过在模型参数上定义网格，然后评估网格上每个点的模型性能（使用验证集（或CV），而不是训练数据）来工作。然后，您可以选择网格上看起来效果最好的点。

示例：在这个任务中，我们不知道max_depth的哪个值最适合应用于波士顿房屋数据集的决策树。我们在max_depth值[1,2，...，10]上定义了一个网格，并为从1到10的每个值评估了决策树的性能。这使我们能够估计max_depth和得分之间的关系。我们通过对这种关系的分析，寻找一个很好的max_depth值，增加它的值不会显着提高。

## 问题 8- 交叉验证
k折交叉验证：我们可以将数据分成用于训练的训练集和用于评估模型性能的测试集。例如，上面使用的测试程序（train_test_split），这是浪费，因为我们使用一部分数据仅用于训练，另一部分用于测试。我们可以通过将数据划分为k块，然后，我们一个接一个地遍历k个块，并使用当前块进行模型验证，剩余的k-1个块用于训练。我们最终得到的是经过训练和评估的模型，我们使用了整个数据集进行验证。

好处是我们可以更可靠地估计网格搜索期间各种参数的配置（使方差较小）。在网格搜索中，存在过度拟合验证集的危险，因为我们多次使用它来评估网格上的不同点的性能并选择提供较好性能的点。因此，随着越来越多的网格点，我们越来越可能找到一个只是偶然好的点。通过交叉验证，过拟合问题得到缓解，因为我们的有效验证集大小更大。

## 练习：训练模型
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

## 问题 9- 最优模型
	# Fit the training data to the model using grid search
	reg = fit_model(X_train, y_train)
	
	# Produce the value for 'max_depth'
	print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))
程序运行结果：

Parameter 'max_depth' is 4 for the optimal model.

**一致的**

## 问题 10 - 预测销售价格
程序运行结果：

- Predicted selling price for Client 1's home: $391,183.33
- Predicted selling price for Client 2's home: $189,123.53
- Predicted selling price for Client 3's home: $942,666.67

误差曲线：

	y_prediction = reg.predict(X_test)
	vs.ErrorFigure(y_prediction, y_test)
曲线绘制代码：

	def ErrorFigure(y_prediction, y_test):
	    # 误差计算
	    allError = []
	    for i in range(0, len(y_test)):
	        error = list(y_test)[i] - list(y_prediction)[i]
	        allError.append(error)
	        # print("The error for {}'s home: ${:,.2f}".format(i, error))
	    # 创建一幅图
	    pl.figure(figsize=(15, 7))
	    # 画出曲线
	    pl.plot(range(1, 99), y_prediction, color='r', label='y_prediction')
	    pl.plot(range(1, 99), y_test, color='g', label='y_test')
	    pl.bar(range(1, 99), allError, alpha=.5, color='b', label='error')
	    pl.legend(loc='upper right')
	    pl.xlabel('Number')
	    pl.ylabel('Price')
	    # 显示
	    pl.show()
![](https://raw.githubusercontent.com/hfrommane/BostonHousing/master/figure/figure_3.png)

灵敏度，程序运行结果：

- Trial 01: $391,183.33
- Trial 02: $424,935.00
- Trial 03: $415,800.00
- Trial 04: $420,622.22
- Trial 05: $418,377.27
- Trial 06: $411,931.58
- Trial 07: $399,663.16
- Trial 08: $407,232.00
- Trial 09: $351,577.61
- Trial 10: $413,700.00

- **Range in prices: $73,357.39**

## 问题11 -适用性
- 近40年前收集的数据与今天的价格无关，因为市场发生变化，房价和货币价值等其他因素也将发生变化。
- 每个家庭提供的数据不足以在当今市场中做出准确的预测。例如，您可以开始考虑诸如该地区的公园或城市中的大学的功能，以更好地预测房子的价格。
- 该模型在特定时间点使用来自某个城市的数据进行训练，因此该模型可能会不及来自不同类型城市（例如农村城市）的变量数据，如问题1的答案中所讨论的。