import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = 'subject2.csv' #subject.csv에 맞게 바꾸기 !!!

df = pd.read_csv(path)
df["location"].describe() # 실내환경인지 확인 !!!

#Unnamed: 0열 제거하기(csv파일을 만들 때 생긴 열)
df = df.drop(df.columns[0], axis=1)

#################################
# 1. 피처별 분석
df.columns
#ColdSens column matplotlib으로 그리기
df["ColdSens"]
plt.plot(df["ColdSens"])
plt.plot(df["ColdExp"])