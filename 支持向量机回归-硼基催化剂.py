import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
# 导入数据
data=pd.read_excel(r"C:\Users\Lenovo\Desktop\机器学习大创项目\中期\描述符-硼基催化剂.xlsx")
# data=data.fillna(0)
data=data.drop(['序号','催化剂引用的文献','催化剂种类','金属1电负性','金属2电负性','非金属1电负性',
                '非金属2电负性','非金属3电负性','负载1','负载电负性'],axis=1)
# 删除缺失值
# data=data.dropna(axis=0,subset=['C2-C3烯烃选择性'])
# print(data.info)
# 填补缺失值
data['C2-C3烯烃选择性']=data['C2-C3烯烃选择性'].fillna(data['C2-C3烯烃选择性'].mean())
data['表面积']=data['表面积'].fillna(data['表面积'].mean())
data=data.fillna(0)
# 元素编码
# 元素编码
data_1=data.loc[:,['非金属1','非金属2','金属1','金属2','非金属3','负载主要元素']]
data_2=data.drop(['非金属1','非金属2','金属1','金属2','非金属3','负载主要元素'],axis=1)
data_1=pd.get_dummies(data_1)
data=pd.concat([pd.DataFrame(data_1),data_2],axis=1)
# 分训练集和测试集
X=data.iloc[:,0:42]
Y=data.iloc[:,43]
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.2,random_state=90)
for i in [Xtrain,Xtest,Ytrain,Ytest]:
    i.index=range(i.shape[0])

# # 恢复索引
# for i in [Xtrain,Xtest]:
#     i.index=range(i.shape[0])
# 数据建模（Lasso）
reg=SVR().fit(Xtrain,Ytrain)
# yhat=reg.predict(Xtest)
# print(yhat)
# print(yhat.min())
# print(yhat.max())
# print('斜率：',reg.coef_.mean())
# print('截距：',reg.intercept_)
# print(zip(Xtrain.columns,reg.coef_))
# 模型评估
# 在模型评估中常常使用MSE、MAE、R方来评价模型数据
# print('MSE:',MSE(yhat,Ytest))
# MSE
print('MSE交叉验证：',cross_val_score(reg,X,Y,cv=8,scoring='neg_mean_squared_error').mean())
# MAE
print('MAE交叉验证：',cross_val_score(reg,X,Y,cv=8,scoring='neg_mean_absolute_error').mean())
# R方
# r2_score_1=reg.score(Xtrain,Ytrain)
# print(r2_score_1)
# 或者通过metrics(前面是真实值，后面是预测值)
# print(r2_score(yhat,Ytest))
print('r2交叉验证：',cross_val_score(reg,X,Y,cv=8,scoring='r2').mean())
