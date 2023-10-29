import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = 'raw_data_Liu.csv'

df = pd.read_csv(path)
df["ID"].unique()

subject3 = df[df["ID"]==3]
subject3

# 1.개인별 최적 온도 찾기를 위해 데이터 개인별로 나누기(14명)
for i in df["ID"].unique():
    subject = df[df["ID"]==i]
    subject.to_csv("subject"+str(i)+".csv")