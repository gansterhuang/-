import pandas as pd
import xgboost as xgb

# 读训练数据

data= pd.read_csv(r'D:\我的文件夹\python_code\Titanic_data\train.csv')


# 数据预处理
data['Sex'] = data['Sex'].apply(lambda s: 1 if s == 'male' else 0)  # 把性别从字符串类型转换为0或1数值型数据
data = data.fillna(0)  # 缺失字段填0
# 选取特征
X_train = data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']].as_matrix()
# 字段说明：性别，年龄，客舱等级，兄弟姐妹和配偶在船数量，父母孩子在船的数量，船票价格

# 建立标签数据集
y_train = data['Survived']

# 训练模型
model = xgb.XGBClassifier(booster='gbtree',max_depth=10, n_estimators=300, learning_rate=0.01).fit(X_train, y_train)



# 读测试数据
testdata = pd.read_csv(r'D:\我的文件夹\python_code\Titanic_data\test.csv')

# 数据清洗, 数据预处理
testdata = testdata.fillna(0)
testdata['Sex'] = testdata['Sex'].apply(lambda s: 1 if s == 'male' else 0)

# 特征选择
X_test = testdata[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']].as_matrix()

# 评估模型
predictions = model.predict(X_test)



# 保存预测结果
submission = pd.DataFrame({'PassengerId': testdata['PassengerId'],
                           'Survived': predictions})
submission.to_csv("titanic_xgboost_submission.csv", index=False)







