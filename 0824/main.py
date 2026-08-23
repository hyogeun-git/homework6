import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import seaborn as sns
import matplotlib.pyplot as plt


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print(train.head())

features = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

missing_info = pd.DataFrame({
    '결측치_개수': train[features].isnull().sum()}
)
print(missing_info)

num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']

for col in num_cols:
    median_val = train[col].median()
    train[col] = train[col].fillna(median_val)
    test[col] = test[col].fillna(median_val)

for col in cat_cols:
    mode_val = train[col].mode()[0]
    train[col] = train[col].fillna(mode_val)
    test[col] = test[col].fillna(mode_val)

missing_info = pd.DataFrame({
    '결측치_개수': train[features].isnull().sum()}
)
print(missing_info)

X_train = pd.get_dummies(train[features])
X_test = pd.get_dummies(test[features])

y_train = train['Transported']


model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

submission = pd.read_csv('data/sample_submission.csv')

submission['Transported'] = prediction

submission.to_csv('my_submission.csv', index=False)

'''--------------------------------------------------'''

x = 'tips'
df = sns.load_dataset(x)

print(df.head())

fig, axes = plt.subplots(2, 3, figsize=(18, 5))

sns.scatterplot(data=df, x='total_bill', y='tip', ax=axes[0, 0])
axes[0, 0].set_title('1. Total Bill vs Tip')

sns.boxplot(data=df, x='time', y='tip', ax=axes[0, 1])
axes[0, 1].set_title('2. Tip by Time (Lunch vs Dinner)')

sns.barplot(data=df, x='day', y='total_bill', ax=axes[0, 2])
axes[0, 2].set_title('3. Total Bill by Day')

sns.scatterplot(data=df, x='total_bill', y='tip', hue='smoker', ax=axes[1, 0])
axes[1, 0].set_title('4. Total Bill vs Tip (by Smoker)')

sns.boxplot(data=df, x='size', y='tip', ax=axes[1, 1])
axes[1, 1].set_title('5. Tip by Party Size')

sns.barplot(data=df, x='sex', y='total_bill', ax=axes[1, 2])
axes[1, 2].set_title('6. Mean Total Bill by Sex')

plt.tight_layout()
plt.show()

'''
e.g. tips

팁에 영향을 준건 식사금액, 방문인원, 방문시간대, 흡연여부, 성별

식사금액 (total_bill) : 고액 결제 > 저액 결제 (가장 결정적 요인)

방문인원 (size) : 4인 이상 단체 > 2인 이하

방문시간대 (time / day) : 주말 저녁 > 평일 점심

흡연여부 (smoker) : 흡연자 (팁 액수 기복 큼, 고액 팁 발생률 높음) > 비흡연자 (안정적, 평균적)

성별 (sex) : 교란변수. 사실 의미 없었음. 식사금액(total_bill) 크기 차이 때문에 우연히 남성이 높아 보였을 뿐 (결제 금액 대비 팁 비율은 동일)

주말 저녁에 온 흡연하는 남자 단체 손님의 양을 늘리면 팁도 늘어날 것이다


'''