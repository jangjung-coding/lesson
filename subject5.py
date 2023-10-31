import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path = 'subject5.csv' #subject.csv에 맞게 바꾸기 !!!

df = pd.read_csv(path)
df["location"].describe() # 실내환경인지 확인 !!!

#Unnamed: 0열 제거하기(csv파일을 만들 때 생긴 열)
df = df.drop(df.columns[0], axis=1)

#################################
sns.set(font_scale=1.1) ## 폰트사이즈 조절
sns.set_style('ticks') ## 축 눈금 표시
data = df[['mean.hr_5', 'mean.WristT_5', 'mean.PantT_5', 'therm_sens', 'therm_pref']]
sns.pairplot(data, diag_kind=None)
plt.show()

sns.set(font_scale=1.1) ## 폰트사이즈 조절
sns.set_style('ticks') ## 축 눈금 표시
data = df[['Sex', 'Age', 'Height', 'Weight', 'therm_sens', 'therm_pref']]
sns.pairplot(data, diag_kind=None)
plt.show()

sns.set(font_scale=1.1) ## 폰트사이즈 조절
sns.set_style('ticks') ## 축 눈금 표시
data = df[['Sex', 'Age', 'Height', 'Weight', 'mean.hr_5', 'mean.WristT_5', 'mean.PantT_5']]
sns.pairplot(data, diag_kind=None)
plt.show()


df[['Sex','therm_sens']].groupby('Sex').boxplot(figsize=(12,12))
plt.show()



