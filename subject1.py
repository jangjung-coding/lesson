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
plt.plot(df['mean.Temperature_60'], color='blue')
plt.plot(df['mean.Temperature_480'], color='black')

plt.plot(df['grad.Temperature_60'], color='red')
plt.plot(df['grad.Temperature_480'], color='black')

plt.plot(df['sd.Temperature_60'], color='green')
plt.plot(df['sd.Temperature_480'], color='black')
############Humidity############
plt.plot(df['mean.Humidity_60'], color='plum')
plt.plot(df['mean.Humidity_480'], color='black')

plt.plot(df['grad.Humidity_60'], color='plum')
plt.plot(df['grad.Humidity_480'], color='black')

plt.plot(df['sd.Humidity_60'], color='plum')
plt.plot(df['sd.Humidity_480'], color='black')
############Winvel############
plt.plot(df['mean.Winvel_60'], color='plum')
plt.plot(df['mean.Winvel_480'], color='black')

plt.plot(df['grad.Winvel_60'], color='plum')
plt.plot(df['grad.Winvel_480'], color='black')

plt.plot(df['sd.Winvel_60'], color='plum')
plt.plot(df['sd.Winvel_480'], color='black')
############Solar############
plt.plot(df['mean.Solar_60'], color='plum')
plt.plot(df['mean.Solar_480'], color='black')

plt.plot(df['grad.Solar_60'], color='plum')
plt.plot(df['grad.Solar_480'], color='black')

plt.plot(df['sd.Solar_60'], color='plum')
plt.plot(df['sd.Solar_480'], color='black')
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
############PantT############
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
plt.plot(df['mean.act_5'], color='black')
plt.plot(df['grad.act_5'], color='red')
plt.plot(df['sd.act_5'], color='blue')

plt.plot(df['mean.act_15'], color='black')
plt.plot(df['grad.act_15'], color='red')
plt.plot(df['sd.act_15'], color='blue')

plt.plot(df['mean.act_60'], color='black')
plt.plot(df['grad.act_60'], color='red')
plt.plot(df['sd.act_60'], color='blue')
#전반적으로 거의다 시간에 상관없이 같은 종류의 피처는 비슷한 양상을 보임. 즉 시간기준으로 데이터를 나누어 학습시켜도 될듯

#Null값 확인
df.isnull().sum()

#피처별 상관관계 그림
lili =list(df.columns)
sns.set(style='whitegrid')
sns.pairplot(df[lili])


