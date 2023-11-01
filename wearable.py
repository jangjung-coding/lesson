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
# 0.개인별 최적 온도 찾기를 위해 데이터 개인별로 나누기(14명)
for i in df_indoor["ID"].unique():
    subject = df_indoor[df_indoor["ID"]==i]
    subject.to_csv("subject"+str(i)+".csv")
    
# 1.피쳐 선별
'''
1.1.피쳐간 상관관계 분석을 위한 산점도 시각화 및 선별
	조건 1) 공간을 실내 오피스로 설정 -> 실외 데이터 제외, 손목 act피쳐 제외
    조건 2) 스마트워치로 쉽게 얻을 수 있는 정보 위주로 -> HR, Wrist
    조건 3) 초기 General model에서는 많은 사람들의 데이터를 한번에 훈련시켜서 일반화를 하므로 개인 인적사항을 우선 배제함
			(뒤에 구성원 사이의 최적 온도를 찾기 위한 Last Model에서는 개인 특성을 더 고려하므로 인적사항 피쳐 사용)
'''
'''
1.2.데이터와 상과도를 보이는 피쳐 추가
	추가 1) Wrist와 양의관계를 보여주는 Ankle 데이터를 추가 -> 데이터 확보로 오버피팅 방지
    추가 2) 실내온도를 보여주는 유일한 피쳐인 PantT 추가
'''
'''
1.3.만족도를 표현하는 피쳐 선별
	피쳐 1) thermal_sens, thermal_pref 사용자의 만족도를 표현하므로
    피쳐 2) ColdSens, ColdExp 질문을 통해 사용자의 체질을 고려할 수 있어서
'''
'''
1.4.피처 최종 선별
    객관적인 피쳐 - mean.hr_60, mean.WristT_60, mean.AnkleT_60, mean.PantT_60
    주관적인 피쳐 - 'ColdSens', 'ColdExp’, 'therm_sens', 'therm_pref’
'''
# 2.선별된 피쳐사이의 관계를 고려해 train, test할 (X,Y)로 변환. 즉, f(피쳐1, 피쳐2, 피쳐3..) = X
'''
2.1. f(mean.hr_60, mean.WristT_60, mean.AnkleT_60, mean.PantT_60) = X
	네가지 피쳐를 하나의 숫자 X로 변환
'''
'''
2.2. f('ColdSens', 'ColdExp’, 'therm_sens', 'therm_pref’) = Y
	네가지 피쳐를 하나의 숫자 Y로 변환
'''
'''
2.3. 기존 피쳐들을 통해 나온 X, Y 시각화
'''
# 3.데이터 수집
# 4.모델 학습 및 성능 평가
'''
4.1. 모델 1 학습 + 성능 평가 
4.2. 모델 2 학습 + 성능 평가 
4.3. 모델 3 학습 + 성능 평가 
4.4. 모델 4 학습 + 성능 평가 
4.5. 모델 5 학습 + 성능 평가 
'''
'''
4.6. General 모델 선정 및 시각화
'''
# 5.모델에 가중치(인적사항)를 주며 14명의 실험체로 개인 range를 잘 찾는지 test하기
'''
5.1. subject1
5.2. 
.
.
.
5.14 subject14
'''
'''
5.15. 시각화 및 성능 평가 발표
'''
# 6.실험실에서 얻은 데이터로 실제상황에서 학습 및 테스트
#

### 두 피쳐간 상관도를 보는 코드(숫자 + 그림)
df['mean.hr_60'].corr(df['mean.PantT_60'])
plt.scatter(df['mean.hr_60'], df['mean.AnkleT_60'], alpha=0.7)
plt.show()

### X를 위한 선별된 피처 산점도(객관 데이터)
sns.set(font_scale=1.1) ## 폰트사이즈 조절
sns.set_style('ticks') ## 축 눈금 표시
data = df[[
        'mean.hr_60', 
        'mean.WristT_60',
        'mean.AnkleT_60', 
        'mean.PantT_60']]
sns.pairplot(data, diag_kind=None,  plot_kws={'alpha':0.2})
plt.show()

### X를 위한 선별된 피처 산점도(객관+주관 데이터)
sns.set(font_scale=1.1)# 폰트사이즈 조절
sns.set_style('whitegrid')# 스타일 설정
data = df[[ # 데이터 선택
    'ColdSens', 'ColdExp',
    'mean.hr_60', 
    'mean.WristT_60',
    'mean.AnkleT_60', 
    'mean.PantT_60']]
sns.pairplot(data, diag_kind='kde', markers='o', plot_kws={'alpha':0.4})# Pairplot 생성
plt.suptitle("Pairplot of Weather Variables", y=1.02)# 그래프 제목 추가
plt.show()