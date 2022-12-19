# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv('/Users/savithakamalon/Desktop/MachineLearning/HousingPriceData/train.csv')
df.head()

df.isnull().sum()
sns.heatmap(df.isnull(), yticklabels=False, cbar= False)


df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())
df.drop(['Alley'],axis = 1, inplace = True)
df['BsmtCond'] = df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtQual'] = df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df['FireplaceQu'] = df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
df['GarageType'] = df['GarageType'].fillna(df['GarageType'].mode()[0])
df.drop(['GarageYrBlt'],axis = 1, inplace = True)
df['GarageFinish'] = df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['GarageQual'] = df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageCond'] = df['GarageCond'].fillna(df['GarageCond'].mode()[0])
df.drop(['PoolQC','Fence','MiscFeature'], axis = 1, inplace=True)

df.drop(['Id'], axis = 1, inplace=True)

df['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])
sns.heatmap(df.isnull(), yticklabels=False, cbar= False)

df['BsmtExposure'] = df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])
df['BsmtFinType2'] = df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])
sns.heatmap(df.isnull(), yticklabels=False, cbar= False, cmap='YlGnBu')

df.dropna(inplace=True)

sns.heatmap(df.isnull(), yticklabels=False, cbar= False, cmap='YlGnBu')

cols= ['MSZoning','Street', 'LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition']
main_df = df.copy()
test_df = pd.read_csv('HousingPriceData/newtest.csv')
final_df = pd.concat([df,test_df],axis =0)

def cat_onehot_multicols(multicols):
    df_final = final_df
    i = 0
    for f in multicols:
        print(f)
        df1 = pd.get_dummies(final_df[f], drop_first = True)
        final_df.drop([f], axis =1 , inplace =True)
        if i==0:
            df_final = df1.copy()
        else : 
            df_final = pd.concat([df_final,df1],axis =1)
        i = i+1
    df_final = pd.concat([final_df,df_final], axis = 1)
    return df_final


final_df = cat_onehot_multicols(cols)

#fin_df = pd.concat([dfcopy, df],axis = 0)
final_df = final_df.loc[:,~final_df.columns.duplicated()]
final_df['SalePrice']

df_Train = final_df.iloc[:1414,:]
df_Test = final_df.iloc[1414:,:]
df_Test.drop(['SalePrice'], axis =1, inplace = True)

X_train = df_Train.drop(['SalePrice'], axis =1)
y_train = df_Train['SalePrice']

import xgboost
classifier = xgboost.XGBRegressor()
classifier.fit(X_train,y_train)

import pickle
filename = 'finalxgboost_model.pkl'
pickle.dump(classifier,open(filename,'wb'))
y_pred = classifier.predict(df_Test)

pred = pd.DataFrame(y_pred)
sub_df = pd.read_csv('HousingPriceData/sample_submission.csv')
datasets = pd.concat([sub_df['Id'],pred], axis = 1)
datasets.columns = ['Id', 'SalePrice']
datasets.to_csv('HousingPriceData/sample_submission_final.csv', index=False)



