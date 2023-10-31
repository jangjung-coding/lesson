import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path = 'raw_data_Liu.csv'

df = pd.read_csv(path)
df["ID"].unique()
df

# 0.실내 환경만 남겨두기 Location: (1 = indoor; -1 = outdoor)
df[df["location"]==1] #3297
df[df["location"]==-1] #546

df_indoor = df[df["location"]==1]

df_indoor

# 1.개인별 최적 온도 찾기를 위해 데이터 개인별로 나누기(14명)
for i in df_indoor["ID"].unique():
    subject = df_indoor[df_indoor["ID"]==i]
    subject.to_csv("subject"+str(i)+".csv")


sns.set(font_scale=1.1) ## 폰트사이즈 조절
sns.set_style('ticks') ## 축 눈금 표시
data = df[['mean.hr_5', 'mean.WristT_5', 'mean.PantT_5', 'therm_sens', 'therm_pref']]
sns.pairplot(data, diag_kind=None)
plt.show()


sns.set(font_scale=1.1) ## 폰트사이즈 조절
sns.set_style('ticks') ## 축 눈금 표시
data = df[['Sex', 'Age', 'Height', 'Weight', 'mean.hr_5', 'mean.WristT_5', 'mean.PantT_5']]
sns.pairplot(data, diag_kind=None)
plt.show()

sns.set(font_scale=1.1) ## 폰트사이즈 조절
sns.set_style('ticks') ## 축 눈금 표시
data = df[['mean.PantT_5','therm_pref']]
sns.pairplot(data, diag_kind=None)
plt.show()

#'mean.PantT_5'를 가로,'therm_pref'를 세로축으로  plot 그려기
plt.plot(df['mean.PantT_5'], color='black')
plt.plot(df['therm_pref'], color='blue')

df['therm_sens'].head(100).plot.line(color='blue', alpha=0.5, label='therm_sens')
df['mean.PantT_5'].head(100).plot.line(color='black', alpha=0.5, label='mean.PantT_5')

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


df['mean.WristT_60'].corr(df['mean.AnkleT_60'])
df['mean.hr_60'].corr(df['mean.PantT_60'])
plt.scatter(df['mean.hr_60'], df['mean.AnkleT_60'], alpha=0.7)
plt.show()

plt.scatter(df['mean.hr_60'], df['mean.PantT_60'], alpha=0.1)
plt.show()


sns.set(font_scale=1.1) ## 폰트사이즈 조절
sns.set_style('ticks') ## 축 눈금 표시
data = df[[
        'mean.hr_60', 
        'mean.WristT_60',
        'mean.AnkleT_60', 
        ]]

sns.pairplot(data, diag_kind='kde', markers='o', plot_kws={'alpha':0.1})
plt.show()

sns.set(font_scale=1.1) ## 폰트사이즈 조절
sns.set_style('ticks') ## 축 눈금 표시
data = df[[
        'mean.hr_60', 
        'mean.WristT_60',
        'mean.AnkleT_60', 
        'mean.PantT_60']]
sns.pairplot(data, diag_kind=None,  plot_kws={'alpha':0.2})
plt.show()

# 폰트사이즈 조절
sns.set(font_scale=1.1)

# 스타일 설정
sns.set_style('whitegrid')

# 데이터 선택
data = df[[
    'ColdSens', 'ColdExp',
    'mean.hr_60', 
    'mean.WristT_60',
    'mean.AnkleT_60', 
    'mean.PantT_60']]

# Pairplot 생성
sns.pairplot(data, diag_kind='kde', markers='o', plot_kws={'alpha':0.4})

# 그래프 제목 추가
plt.suptitle("Pairplot of Weather Variables", y=1.02)

# 그래프 보이기
plt.show()

df.columns

df.columns