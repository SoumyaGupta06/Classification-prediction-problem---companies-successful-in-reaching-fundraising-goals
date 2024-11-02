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
