import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, 
                             classification_report, roc_auc_score, log_loss)
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Load data
data_ico = pd.read_csv("D:\LUBS5990M_Data_202223.csv", encoding='UTF-8')
print(data_ico.describe())
print(data_ico.shape)

#----------------------------Data preprocessing----------------------------------
dateDel = data_ico
dateDel = dateDel[(dateDel['distributedPercentage'] > 0) & (dateDel['distributedPercentage'] <= 1)]

#----Outliers----
data_cleaning = dateDel
# Detect and remove outliers for 'coinNum'
coin_outliers = data_cleaning['coinNum'][((data_cleaning['coinNum'] < data_cleaning['coinNum'].quantile(0.25) - 1.5 * (data_cleaning['coinNum'].quantile(0.75) - data_cleaning['coinNum'].quantile(0.25))) | 
                                          (data_cleaning['coinNum'] > data_cleaning['coinNum'].quantile(0.75) + 1.5 * (data_cleaning['coinNum'].quantile(0.75) - data_cleaning['coinNum'].quantile(0.25))))]
coin_no_outlier = data_cleaning[~data_cleaning['coinNum'].isin(coin_outliers)]
plt.boxplot(coin_no_outlier['coinNum'])
plt.ylabel("coinNum")
plt.show()

plt.hist(coin_no_outlier['coinNum'], bins=int(np.sqrt(len(coin_no_outlier))))
plt.xlabel("coinNum")
plt.title("Histogram-coinNum (without outliers)")
plt.show()

# Detect and remove outliers for 'priceUSD'
price_outliers = coin_no_outlier['priceUSD'][((coin_no_outlier['priceUSD'] < coin_no_outlier['priceUSD'].quantile(0.25) - 1.5 * (coin_no_outlier['priceUSD'].quantile(0.75) - coin_no_outlier['priceUSD'].quantile(0.25))) | 
                                              (coin_no_outlier['priceUSD'] > coin_no_outlier['priceUSD'].quantile(0.75) + 1.5 * (coin_no_outlier['priceUSD'].quantile(0.75) - coin_no_outlier['priceUSD'].quantile(0.25))))]
coin_no_outlier2 = coin_no_outlier[~coin_no_outlier['priceUSD'].isin(price_outliers)]
plt.boxplot(coin_no_outlier2['priceUSD'])
plt.ylabel("priceUSD")
plt.show()

plt.hist(coin_no_outlier2['priceUSD'], bins=int(np.sqrt(len(coin_no_outlier2))))
plt.xlabel("priceUSD")
plt.title("priceUSD (without outliers)")
plt.show()

coin_no_outlier.to_csv("D:\CleanDataset(without_outliers).csv", index=False)

#----Missing Data----

data_imput = pd.read_csv("D:\CleanDataset(without_outliers).csv", encoding='UTF-8')
print(data_imput.isna().sum())  

numeric_data = data_imput.select_dtypes(include=[np.number])
non_numeric_data = data_imput.select_dtypes(exclude=[np.number])

imputer = IterativeImputer(max_iter=10, random_state=0)
numeric_data_imputed = pd.DataFrame(imputer.fit_transform(numeric_data), columns=numeric_data.columns)

data_imputed = pd.concat([numeric_data_imputed, non_numeric_data.reset_index(drop=True)], axis=1)

data_imputed.to_csv("D:\MLDataset.csv", index=False)

print("Final shape after combining:", data_imputed.shape)


#------------------Model development------------------------

# model evaluation
def evaluate_model(y_test, y_pred, y_pred_proba=None, multi_class=False):
    metrics = {
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Accuracy Score": accuracy_score(y_test, y_pred),
        "Classification Report": classification_report(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_pred_proba if y_pred_proba is not None else y_pred, 
                                 multi_class='ovr' if multi_class else 'raise'),
        "Log Loss": log_loss(y_test, y_pred_proba if y_pred_proba is not None else y_pred)
    }
    for metric, value in metrics.items():
        print(f"{metric}:\n{value}\n")

# Load and prepare data
data = pd.read_csv(r'D:\MLDataset.csv')
features = ["hasVideo", "rating", "priceUSD", "Region.US", "teamSize", 
            "hasGithub", "hasReddit", "platformIndicator", "coinNum", 
            "minInvestment", "distributedPercentage", "durationDays"]
X = data[features]
y = data["successIndicator"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes Model (Categorical Features Only)
print("Naive Bayes Results:")
categorical_features = ["hasVideo", "Region.US", "hasGithub", "hasReddit", 
                        "platformIndicator", "minInvestment"]
X_train_cat = X_train[categorical_features]
X_test_cat = X_test[categorical_features]

cnb = CategoricalNB()
cnb.fit(X_train_cat, y_train)
categorical_pred = cnb.predict(X_test_cat)
evaluate_model(y_test, categorical_pred)

# Decision Tree with AdaBoost
print("Decision Tree with AdaBoost Results:")
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1],
    'algorithm': ['SAMME', 'SAMME.R']
}
adb_classifier = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3, random_state=1), random_state=1)
grid_search = GridSearchCV(adb_classifier, param_grid, scoring='accuracy', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

y_pred_adb = grid_search.best_estimator_.predict(X_test)
evaluate_model(y_test, y_pred_adb)


# Artificial Neural Network (ANN)
print("ANN Results:")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

y_pred = model.predict(X_test_scaled)
y_pred_ann = (y_pred > 0.5).astype(int)
evaluate_model(y_test, y_pred_ann, y_pred)
