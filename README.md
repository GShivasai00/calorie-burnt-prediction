# calorie-burnt-prediction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
df1 = pd.read_csv('calories.csv')
df1.head()
df2 = pd.read_csv('exercise.csv')
df2.head()
df3 = pd.concat([df2,df1['Calories']], axis=1)
df3.head()
df3.shape
df3.info()
df3.describe()
sb.scatterplot(x='Height', y='Weight', data=df3)
plt.show()
features = ['Age', 'Height', 'Weight', 'Duration']

plt.subplots(figsize=(15, 10))
for i, col in enumerate(features):
    plt.subplot(2, 2, i + 1)
    # Instead of sampling, use the original DataFrame 'df' with a condition to limit the data points.
    # This will ensure that 'Calories' column is present.
    x = df3.sample(1000)  # Keep this line if you want a random sample
    sb.scatterplot(x=col, y='Calories', data=df3[df3.index.isin(x.index)])
plt.tight_layout()
plt.show()
features = df3.select_dtypes(include='float').columns

plt.subplots(figsize=(15, 10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sb.displot(df3[col])
plt.tight_layout()
plt.show()
df3.replace({'male': 0, 'female': 1},
           inplace=True)
df3.head()
plt.figure(figsize=(8, 8))
sb.heatmap(df3.corr() > 0.9,
           annot=True,
           cbar=False)
plt.show()
features = df3.drop(['User_ID', 'Calories'], axis=1)
target = df3['Calories'].values

X_train, X_val,\
    Y_train, Y_val = train_test_split(features, target,
                                      test_size=0.1,
                                      random_state=22)
X_train.shape, X_val.shape
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)
print(X.shape,X_train.shape,X_test.shape
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
from sklearn.metrics import mean_absolute_error as mae
models = [LinearRegression(), XGBRegressor(),
Lasso(), RandomForestRegressor(), Ridge()]

for i in range(5):
    models[i].fit(X_train, Y_train)
    print(f'{models[i]} : ')
    train_preds = models[i].predict(X_train)
    print('Training Error : ', mae(Y_train, train_preds))
    val_preds = models[i].predict(X_val)
    print('Validation Error : ', mae(Y_val, val_preds))
    print()
    
model = XGBRegressor()
#training the model with X_train
model.fit(X_train,Y_train)
# Assuming you want to predict using the last trained model (XGBRegressor), you can do:
calories_burnt_prediction = models[-1].predict(X_test)
# models[-1] accesses the last element of the models list which should be your XGBRegressor
print(calories_burnt_prediction)

# Or, if you want to predict using each model in the list, you would need to loop through it:
for model in models:
    calories_burnt_prediction = model.predict(X_test)
    print(f"Prediction from {type(model).__name__}: {calories_burnt_prediction}")
MAE = metrics.mean_absolute_error(Y_test, calories_burnt_prediction)
print("Mean Absolute Error = ",MAE)
from sklearn.metrics import mean_absolute_error

val_preds = model.predict(X_val)
# Use mean_absolute_error directly after importing it
mae_xgb = mean_absolute_error(Y_val, val_preds)
print("MAE for XGBoost:", mae_xgb)
input_data = (0,68,190.0,94.0,29.0,105.0,40.8,231)
#input_data_as_numpy_array = np.asarray(input_data)
#input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
#prediction = model.predict(input_data_reshaped)
#print(prediction)
#print("The initial value is ",prediction[0])
print("The calories burnt for the first individual in the dataset is predicted as ", calories_burnt_prediction[0])
print("Thus we have successfully predicted the calories burnt using XGBoost")





