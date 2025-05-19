import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, 
                             classification_report, roc_auc_score, log_loss)
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.stats import ttest_ind, chi2_contingency, f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Load data
df = pd.read_csv("D:\original_data.csv", encoding='latin1')
print(df.describe())
print(df.shape)
print(df.head())

#----------------------------Data preprocessing----------------------------------

df = df[(df['distributedPercentage'] > 0) & (df['distributedPercentage'] <= 1)]
df['platform'] = df['platform'].str.strip()

#----Outliers----
data_cleaning = df
# Detect and remove outliers for 'coinNum'
lbound = data_cleaning['coinNum'].quantile(0.25) - 1.5 * (data_cleaning['coinNum'].quantile(0.75) - data_cleaning['coinNum'].quantile(0.25))
ubound = data_cleaning['coinNum'].quantile(0.75) + 1.5 * (data_cleaning['coinNum'].quantile(0.75) - data_cleaning['coinNum'].quantile(0.25))
coin_outliers = data_cleaning.loc[(data_cleaning['coinNum'] < lbound) | (data_cleaning['coinNum'] > ubound)].index
coin_no_outlier = data_cleaning.drop(index=coin_outliers)
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

coin_no_outlier2.to_csv("D:\CleanDataset(without_outliers).csv", index=False)

#----Missing data----

data_imput = pd.read_csv("D:\CleanDataset(without_outliers).csv", encoding='UTF-8')
print(data_imput.isna().sum())  

numeric_data = data_imput.select_dtypes(include=[np.number])
non_numeric_data = data_imput.select_dtypes(exclude=[np.number])

imputer = IterativeImputer(max_iter=10, random_state=0)
numeric_data_imputed = pd.DataFrame(imputer.fit_transform(numeric_data), columns=numeric_data.columns)

data_imputed = pd.concat([numeric_data_imputed, non_numeric_data.reset_index(drop=True)], axis=1)

#------------------------------Transformation---------------------
def reg(countryRegion):
    if countryRegion=='USA':
        return 1
    else:
        return 0

data_imputed['region_indicator'] = data_imputed['countryRegion'].apply(reg)

def dayscalc(row):
    start_date = pd.to_datetime(row['startDate'], format='%d/%m/%Y')
    end_date = pd.to_datetime(row['endDate'], format='%d/%m/%Y')
    if pd.isnull(start_date) or pd.isnull(end_date):
        return -1
    if start_date < end_date:
        return (end_date - start_date).days
    else: 
        return -1

data_imputed['days'] = data_imputed.apply(dayscalc, axis=1)
data_imputed = data_imputed[data_imputed['days']> -1]

def tgtcalc(success):
    if success=='Y':
        return 1
    else:
        return 0
    
data_imputed['success_indicator'] = data_imputed['success'].apply(tgtcalc)
data_imputed['platformIndicator'] = data_imputed['platform'].apply(lambda x: 1 if x in ['Ethereum','Ethereum, Waves','ETH','Etherum','Ethererum'] else 0)
print("Final shape after combining:", data_imputed.shape)

#-----Normalization-----
df = data_imputed
cols_to_scale = ['rating', 'priceUSD', 'teamSize', 'coinNum', 'days']
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
df_scaled.to_csv("D:\MLDataset.csv", index=False)
print(df_scaled)

#--------------EDA-----------------

df = pd.read_csv(r"D:\MLDataset.csv", encoding='latin1')

numerical_cols = ['rating', 'priceUSD', 'teamSize', 'coinNum', 'distributedPercentage', 'days']
categorical_cols = ['hasVideo', 'hasGithub', 'hasReddit', 'minInvestment', 'region_indicator','platformIndicator']
target_col = 'success_indicator'  

vif_check_data = df[['rating', 'priceUSD', 'teamSize', 'coinNum', 'distributedPercentage', 'days', 'hasVideo', 'hasGithub', 'hasReddit', 'minInvestment', 'region_indicator','platformIndicator']]

results = []

# --- T-Test for Continuous Features ---
for col in numerical_cols:
    if col == target_col:
        continue
    try:
        group0 = df[df[target_col].values.ravel() == 0][col].dropna()
        group1 = df[df[target_col].values.ravel() == 1][col].dropna()
        if len(group0) > 1 and len(group1) > 1 and group0.nunique() > 1 and group1.nunique() > 1:
            stat, pval = ttest_ind(group0, group1, equal_var=False)  # Welch's t-test
            results.append((col, 't-test', pval))
        else:
            results.append((col, 't-test', np.nan))
    except Exception as e:
        print(f"Error in t-test for {col}: {e}")
        results.append((col, 't-test', np.nan))

# --- Chi-Squared Test for Categorical Features ---
for col in categorical_cols:
    try:
        if df[col].nunique() <= 20:  # avoid large contingency tables
            contingency_table = pd.crosstab(df[col], df[target_col])
            chi2, pval, _, _ = chi2_contingency(contingency_table)
            results.append((col, 'Chi2', pval))
    except Exception as e:
        print(f"Error in Chi-Squared for {col}: {e}")
        results.append((col, 'Chi2', np.nan))

results_df = pd.DataFrame(results, columns=['Feature', 'Test', 'p-value'])
results_df['p-value'] = results_df['p-value'].map(lambda x: f'{x:.10f}')
print(results_df.sort_values(by='p-value'))

# --- Multicollinearity check ---
vif_data = pd.DataFrame()
vif_data["Feature"] = vif_check_data.columns
vif_data["VIF"] = [variance_inflation_factor(vif_check_data.values, i) for i in range(vif_check_data.shape[1])]
print(vif_data.sort_values(by="VIF", ascending=False))


#------------------Model development------------------------

# Model evaluation
def evaluate_model(name, y_test, y_pred, y_pred_proba=None, multi_class=False):
    metrics = {
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Accuracy Score": accuracy_score(y_test, y_pred),
        "Classification Report": classification_report(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_pred_proba if y_pred_proba is not None else y_pred, 
                                 multi_class='ovr' if multi_class else 'raise'),
        "Log Loss": log_loss(y_test, y_pred_proba if y_pred_proba is not None else y_pred)
    }
    print(f"{name} model results:")
    for metric, value in metrics.items():
        print(f"{metric}:\n{value}\n")

# Load and prepare data
data = pd.read_csv(r'D:\MLDataset.csv')
df = pd.DataFrame(data)
correlation_matrix = df[['priceUSD', 'teamSize', 'coinNum', 'days', 
                         'hasVideo', 'hasGithub', 'hasReddit', 
                         'minInvestment', 'region_indicator']].corr()
print(correlation_matrix)

features = ['priceUSD', 'teamSize', 'coinNum', 'days', 
                         'hasVideo', 'hasGithub', 'hasReddit', 
                         'minInvestment', 'region_indicator']
X = data[features]
y = data["success_indicator"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----SVM---- 
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(probability=True, random_state=42))  # Note: the step name is 'svc'
])
param_grid = {
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['linear', 'rbf'],
    'svc__gamma': ['scale', 'auto'],
    'svc__degree':  [2, 3, 4],                 
    'svc__class_weight': [None, 'balanced']    
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)
y_pred = grid_search.predict(X_test)
evaluate_model('SVM', y_test, y_pred)


#----Decision Tree----
clf = DecisionTreeClassifier(random_state=42, max_depth=5)  
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
evaluate_model('Decision Tree', y_test, y_pred)

#----XGBoost----
xgb_clf = XGBClassifier(n_estimators=300,
                        learning_rate=0.05,
                        max_depth=5,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective='binary:logistic',
                        eval_metric='logloss',
                        random_state=42,
                        n_jobs=-1)
xgb_clf.fit(X_train, y_train)
preds = xgb_clf.predict(X_test)
evaluate_model('XGBoost', y_test, y_pred)

#----Random Forest----
rf_clf = RandomForestClassifier(n_estimators=400,
                                max_depth=None,
                                class_weight='balanced',
                                random_state=42,
                                n_jobs=-1)
rf_clf.fit(X_train, y_train)
preds = rf_clf.predict(X_test)
evaluate_model('Random Forest', y_test, y_pred)

#----Gradient Boosting----
gbm_clf = GradientBoostingClassifier(n_estimators=300,
                                     learning_rate=0.05,
                                     max_depth=3,
                                     random_state=42)
gbm_clf.fit(X_train, y_train)
preds = gbm_clf.predict(X_test)
evaluate_model('GBM', y_test, y_pred)


#----Artificial Neural Network----
print("ANN Results:")
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
y_pred = model.predict(X_test)
y_pred_ann = (y_pred > 0.5).astype(int)
evaluate_model('ANN', y_test, y_pred_ann, y_pred)

