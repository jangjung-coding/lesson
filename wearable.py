import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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