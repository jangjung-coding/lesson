'''
Units
Temperature: ˚C
Wind speed: m/s
Heart rate: bpm
Acceleration: m/s2
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path = 'subject1.csv' #subject.csv에 맞게 바꾸기 !!!

df = pd.read_csv(path)
df["location"].describe() # 실내환경인지 확인

#Unnamed: 0열 제거하기(csv파일을 만들 때 생긴 열)
df = df.drop(df.columns[0], axis=1)

#################################
# 1. 피처별 분석
df.columns
#ColdSens column matplotlib으로 그리기
df["ColdSens"]
plt.plot(df["ColdSens"])

plt.plot(df["ColdExp"])

############Temperature############
#plt.plot(df['mean.Temperature_60'], color='blue')
#plt.plot(df['mean.Temperature_480'], color='black')

#plt.plot(df['grad.Temperature_60'], color='red')
#plt.plot(df['grad.Temperature_480'], color='black')

#plt.plot(df['sd.Temperature_60'], color='green')
#plt.plot(df['sd.Temperature_480'], color='black')
############Humidity############
#plt.plot(df['mean.Humidity_60'], color='plum')
#plt.plot(df['mean.Humidity_480'], color='black')

#plt.plot(df['grad.Humidity_60'], color='plum')
#plt.plot(df['grad.Humidity_480'], color='black')

#plt.plot(df['sd.Humidity_60'], color='plum')
#plt.plot(df['sd.Humidity_480'], color='black')
############Winvel############
#plt.plot(df['mean.Winvel_60'], color='plum')
#plt.plot(df['mean.Winvel_480'], color='black')

#plt.plot(df['grad.Winvel_60'], color='plum')
#plt.plot(df['grad.Winvel_480'], color='black')

#plt.plot(df['sd.Winvel_60'], color='plum')
#plt.plot(df['sd.Winvel_480'], color='black')
############Solar############
#plt.plot(df['mean.Solar_60'], color='plum')
#plt.plot(df['mean.Solar_480'], color='black')

#plt.plot(df['grad.Solar_60'], color='plum')
#plt.plot(df['grad.Solar_480'], color='black')

#plt.plot(df['sd.Solar_60'], color='plum')
#plt.plot(df['sd.Solar_480'], color='black')
############Heart Rate############
plt.plot(df['mean.hr_5'], color='black')
plt.plot(df['grad.hr_5'], color='red')
plt.plot(df['sd.hr_5'], color='blue')

plt.plot(df['mean.hr_15'], color='black')
plt.plot(df['grad.hr_15'], color='red')
plt.plot(df['sd.hr_15'], color='blue')

plt.plot(df['mean.hr_60'], color='black')
plt.plot(df['grad.hr_60'], color='red')
plt.plot(df['sd.hr_60'], color='blue')
############WristT############
plt.plot(df['mean.WristT_5'], color='black')
plt.plot(df['grad.WristT_5'], color='red')
plt.plot(df['sd.WristT_5'], color='blue')

plt.plot(df['mean.WristT_15'], color='black')
plt.plot(df['grad.WristT_15'], color='red')
plt.plot(df['sd.WristT_15'], color='blue')

plt.plot(df['mean.WristT_60'], color='black')
plt.plot(df['grad.WristT_60'], color='red')
plt.plot(df['sd.WristT_60'], color='blue')
############AnkleT############
plt.plot(df['mean.AnkleT_5'], color='black')
plt.plot(df['grad.AnkleT_5'], color='red')
plt.plot(df['sd.AnkleT_5'], color='blue')

plt.plot(df['mean.AnkleT_15'], color='black')
plt.plot(df['grad.AnkleT_15'], color='red')
plt.plot(df['sd.AnkleT_15'], color='blue')

plt.plot(df['mean.AnkleT_60'], color='black')
plt.plot(df['grad.AnkleT_60'], color='red')
plt.plot(df['sd.AnkleT_60'], color='blue')
# ############PantT############
plt.plot(df['mean.PantT_5'], color='black')
plt.plot(df['grad.PantT_5'], color='red')
plt.plot(df['sd.PantT_5'], color='blue')

plt.plot(df['mean.PantT_15'], color='black')
plt.plot(df['grad.PantT_15'], color='red')
plt.plot(df['sd.PantT_15'], color='blue')

plt.plot(df['mean.PantT_60'], color='black')
plt.plot(df['grad.PantT_60'], color='red')
plt.plot(df['sd.PantT_60'], color='blue')
############act############
#plt.plot(df['mean.act_5'], color='black')
#lt.plot(df['grad.act_5'], color='red')
#plt.plot(df['sd.act_5'], color='blue')

#plt.plot(df['mean.act_15'], color='black')
#plt.plot(df['grad.act_15'], color='red')
#plt.plot(df['sd.act_15'], color='blue')

#plt.plot(df['mean.act_60'], color='black')
#plt.plot(df['grad.act_60'], color='red')
#plt.plot(df['sd.act_60'], color='blue')
#전반적으로 거의다 시간에 상관없이 같은 종류의 피처는 비슷한 양상을 보임. 즉 시간기준으로 데이터를 나누어 학습시켜도 될듯

#Null값 확인
print((df.isnull().sum()>=40).T)

#피처별 상관관계 그림
plt.figure(figsize=(15,15))
sns.heatmap(data = df.corr(), annot=True,
fmt = '.2f', linewidths=.5, cmap='Blues')

# mean.Temperature_60를 x축, therm_sens를 y축으로 그림
plt.scatter(df['mean.Temperature_60'], df['therm_sens'])

# therm_pref와 therm_sens의 상관관계 그림. legend로 축이름 지정
plt.scatter(df['therm_pref'], df['therm_sens'], label=('therm_pref', 'therm_sens'))

# mean.Temperature_60를 x축, therm_pref축으로 그림
plt.scatter(df['mean.Temperature_60'], df['therm_pref'])

plt.plot(df["mean.Temperature_60"])
df["mean.Temperature_60"].describe()

plt.plot(df['mean.AnkleT_60'], color='black')
plt.plot(df['mean.PantT_60'], color='blue')
plt.plot(df['mean.WristT_60'], color='red')

# Heart rate 와 WristT의 상관관계
plt.scatter(df['mean.hr_5'], df['mean.WristT_5'])

# Heart rate, Wrist, Solar, Temperature 각각의 산점도 상관관계 그림
plt.scatter(df['mean.hr_5'], df['mean.WristT_5'], label='hr')

# Solar와 Temperature의 상관관계
plt.scatter(df['mean.Solar_60'], df['mean.Temperature_60'])

plt.scatter(df['mean.WristT_5'], df['mean.PantT_5'])
# mean.PantT_5와 mean.WristT_5의 상관관계 수치화
df['mean.PantT_5'].corr(df['mean.WristT_5'])
plt.scatter(df['mean.WristT_15'], df['mean.PantT_15'])
df['mean.PantT_15'].corr(df['mean.WristT_15'])
plt.scatter(df['mean.WristT_60'], df['mean.PantT_60'])
df['mean.PantT_60'].corr(df['mean.WristT_60'])

df.columns

sns.set(font_scale=1.1) ## 폰트사이즈 조절
sns.set_style('ticks') ## 축 눈금 표시
data = df[['Sex', 'Age', 'Height', 'Weight', 'ColdSens', 'ColdExp', 
        'therm_sens', 'therm_pref', 'location',
        'mean.Temperature_60',
        'mean.Humidity_60',
        'mean.Winvel_60', 
        'mean.Solar_60',
        'mean.hr_60', 
        'mean.WristT_60',
        'mean.AnkleT_60', 
        'mean.PantT_60']]
sns.pairplot(data, diag_kind=None)
plt.show()

sns.set(font_scale=1.1) ## 폰트사이즈 조절
sns.set_style('ticks') ## 축 눈금 표시
data = df[[
        'mean.hr_60', 
        'mean.WristT_60',
        'mean.AnkleT_60', 
        'mean.PantT_60',
        'therm_sens']]
sns.pairplot(data, diag_kind=None)
plt.show()

df = df[[
        'mean.hr_60', 
        'mean.WristT_60',
        'mean.AnkleT_60', 
        'mean.PantT_60',
        'therm_sens',
        'therm_pref']]

# 2. 데이터 전처리
# 2-1. 데이터 정규화
#Nan값 확인
df.isnull().sum()
#Nan값 평균으로 대체
df = df.fillna(df.mean())
# 2-1-1. MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(df)
df_scaled = scaler.transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
df_scaled

# 2-1-2. StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
df_scaled = scaler.transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
df_scaled

# 2-2. 데이터 분할
# 2-2-1. train, test 데이터 분할
from sklearn.model_selection import train_test_split
train, test = train_test_split(df_scaled, test_size=0.2, random_state=0)
train.shape
test.shape

# 2-2-2. train, validation 데이터 분할
train, val = train_test_split(train, test_size=0.2, random_state=0)
train.shape
val.shape

# 2-2-3. X, y 분할
X_train = train.drop(columns=['therm_sens', 'therm_pref'])
y_train = train[['therm_sens', 'therm_pref']]
X_val = val.drop(columns=['therm_sens', 'therm_pref'])
y_val = val[['therm_sens', 'therm_pref']]
X_test = test.drop(columns=['therm_sens', 'therm_pref'])
y_test = test[['therm_sens', 'therm_pref']]
X_train.shape
y_train.shape
X_val.shape
y_val.shape
X_test.shape
y_test.shape

# 3. 모델링
# 3-1. 모델링
# 3-1-1. Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_val)
lr_pred

# 3-1-2. Ridge Regression
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_val)
ridge_pred

# 3-1-3. Lasso Regression
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_val)
lasso_pred

# 3-1-4. ElasticNet Regression
from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet()
elasticnet.fit(X_train, y_train)
elasticnet_pred = elasticnet.predict(X_val)
elasticnet_pred

# 3-1-5. Decision Tree
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor()
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_val)
tree_pred

# 3-1-6. Random Forest
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor()
forest.fit(X_train, y_train)
forest_pred = forest.predict(X_val)
forest_pred

# 3-1-7. Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)
gbr_pred = gbr.predict(X_val)
gbr_pred

# 3-1-8. XGBoost
from xgboost import XGBRegressor
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_val)
xgb_pred

# 3-1-9. LightGBM
from lightgbm import LGBMRegressor
lgbm = LGBMRegressor()
lgbm.fit(X_train, y_train)
lgbm_pred = lgbm.predict(X_val)
lgbm_pred

# 3-1-10. CatBoost
from catboost import CatBoostRegressor
cat = CatBoostRegressor()
cat.fit(X_train, y_train)
cat_pred = cat.predict(X_val)
cat_pred

# 3-2. 모델 평가
# 3-2-1. MSE
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_val, lr_pred)
mse

# 3-2-2. RMSE
rmse = np.sqrt(mse)
rmse

# 3-2-3. R2
from sklearn.metrics import r2_score
r2 = r2_score(y_val, lr_pred)
r2

# 3-2-4. MAE
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_val, lr_pred)
mae


# 4. 모델 튜닝
# 4-1. GridSearchCV
from sklearn.model_selection import GridSearchCV
params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [1, 2, 3],
    'learning_rate': [0.1, 0.2, 0.3]
}
grid_cv = GridSearchCV(gbr, param_grid=params, cv=3, scoring='neg_mean_squared_error', verbose=1)
grid_cv.fit(X_train, y_train)
print('최적 하이퍼 파라미터: ', grid_cv.best_params_)
print('최적 평균 RMSE: ', np.sqrt(np.abs(grid_cv.best_score_)))

# 4-2. RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [1, 2, 3],
    'learning_rate': [0.1, 0.2, 0.3]
}
random_cv = RandomizedSearchCV(gbr, param_distributions=params, cv=3, scoring='neg_mean_squared_error', verbose=1)
random_cv.fit(X_train, y_train)
print('최적 하이퍼 파라미터: ', random_cv.best_params_)
print('최적 평균 RMSE: ', np.sqrt(np.abs(random_cv.best_score_)))

# 5. 모델 재학습
# 5-1. Gradient Boosting
gbr = GradientBoostingRegressor(learning_rate=0.2, max_depth=3, n_estimators=300)
gbr.fit(X_train, y_train)
gbr_pred = gbr.predict(X_test)
gbr_pred

# 5-2. XGBoost
xgb = XGBRegressor(learning_rate=0.2, max_depth=3, n_estimators=300)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_pred

# 5-3. LightGBM
lgbm = LGBMRegressor(learning_rate=0.2, max_depth=3, n_estimators=300)
lgbm.fit(X_train, y_train)
lgbm_pred = lgbm.predict(X_test)
lgbm_pred

# 5-4. CatBoost
cat = CatBoostRegressor(learning_rate=0.2, max_depth=3, n_estimators=300)
cat.fit(X_train, y_train)
cat_pred = cat.predict(X_test)
cat_pred

# 6. 모델 평가
# 6-1. MSE
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, gbr_pred)
mse

# 6-2. RMSE
rmse = np.sqrt(mse)
rmse

# 6-3. R2
from sklearn.metrics import r2_score
r2 = r2_score(y_test, gbr_pred)
r2

# 6-4. MAE
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, gbr_pred)
mae

# 7. 모델 저장
import pickle
pickle.dump(gbr, open('gbr.pkl', 'wb'))
pickle.dump(xgb, open('xgb.pkl', 'wb'))
pickle.dump(lgbm, open('lgbm.pkl', 'wb'))
pickle.dump(cat, open('cat.pkl', 'wb'))

# 8. 모델 불러오기
import pickle
gbr = pickle.load(open('gbr.pkl', 'rb'))
xgb = pickle.load(open('xgb.pkl', 'rb'))
lgbm = pickle.load(open('lgbm.pkl', 'rb'))
cat = pickle.load(open('cat.pkl', 'rb'))

# 9. 모델 예측
# 9-1. Gradient Boosting
gbr_pred = gbr.predict(X_test)
gbr_pred

# 9-2. XGBoost
xgb_pred = xgb.predict(X_test)
xgb_pred

# 9-3. LightGBM
lgbm_pred = lgbm.predict(X_test)
lgbm_pred

# 9-4. CatBoost
cat_pred = cat.predict(X_test)
cat_pred

# 10. 모델 평가
# 10-1. MSE
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, gbr_pred)
mse

# 10-2. RMSE
rmse = np.sqrt(mse)
rmse
