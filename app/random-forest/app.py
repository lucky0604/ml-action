import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pydot

features = pd.read_csv('temps.csv')
features.head(5)
'''
one hot 方式将week列进行编码（pandas中使用get_dummies方法）
one-hot方式：
https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f#:~:text=One%20hot%20encoding%20is%20a%20process%20by%20which,numerical%20value%20of%20the%20entry%20in%20the%20dataset.
'''
features = pd.get_dummies(features)
print(features.head(5))
# 靶向量（因变量）
targets = features['actual']

# 从特征矩阵中移除actual这一列，axis = 1表示移除列的方向是列方向
features = features.drop('actual', axis=1)

# 特征名列表
feature_list = list(features.columns)
print(feature_list)

# 将数据分为训练集和测试集
train_features, test_features, train_targets, test_targets = train_test_split(
    features, targets, test_size=0.25, random_state=42)

print(train_features, ' train features')
print(test_features, ' test features')
print(train_targets, ' train targets')
print(test_targets, ' test targets')

# 建立基准线模型
# 选中test_features所有行
# 选中test_features中average列
baseline_preds = test_features.loc[:, 'average']

baseline_errors = abs(baseline_preds - test_targets)
print('平均误差：', round(np.mean(baseline_errors), 2))

# 训练随即森林模型
# 1000个决策树
rf = RandomForestRegressor(n_estimators=1000, random_state=42)
rf.fit(train_features, train_targets)

# 运行结果
RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                      max_features='auto', max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2,
                      min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=1, oob_score=False, random_state=42, verbose=0, warm_start=False)

# 检验模型训练效果
predictions = rf.predict(test_features)
errors = abs(predictions - test_targets)
print('平均误差：', round(np.mean(errors), 2))

# 计算平均绝对百分误差mean absolute percentage error(MAPE)
mape = 100 * (errors / test_targets)

accuracy = 100 - np.mean(mape)
print('准确率：', round(accuracy, 2), '%.')

print('模型中的决策树有：', len(rf.estimators_), ' 个')

# 从1000个决策树中选出前5个
print(rf.estimators_[:5])

# 从1000个决策树中选择第6个
tree = rf.estimators_[5]

# 将决策树输出到dot文件中
export_graphviz(
    tree,
    out_file='tree.dot',
    feature_names=feature_list,
    rounded=True,
    precision=1
)

# 将dot文件转化为图结构
(graph, ) = pydot.graph_from_dot_file('tree.dot')

# 将graph图输出为png图片文件
graph.write_png('tree.png')
