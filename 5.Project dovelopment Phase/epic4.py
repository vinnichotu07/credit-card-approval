import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("🤖 Epic 4: Starting Model Building and Training...")

# 1. Load the cleaned data matrix from Epic 3
df = pd.read_csv('cleaned_credit_record.csv')

# 2. Separate our features (X) from our target prediction answer (y)
# We drop 'Approved' because that's what we want the AI to guess
X = df.drop(columns=['Approved'])
y = df['Approved']

# Split data into training sets (80%) and testing sets (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- MODEL 1: LOGISTIC REGRESSION ---
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr_model.predict(X_test))
print(f" ✅ Logistic Regression Accuracy: {lr_acc * 100}%")

# --- MODEL 2: DECISION TREE ---
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt_model.predict(X_test))
print(f" ✅ Decision Tree Accuracy: {dt_acc * 100}%")

# --- MODEL 3: RANDOM FOREST ---
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
print(f" ✅ Random Forest Accuracy: {rf_acc * 100}%")

# 3. Find the best model and save it physically as 'model.pkl'
best_model = rf_model  # Defaulting to Random Forest as it typically handles mixed data best
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("\n🎉 EPIC 4 SUCCESSFUL!")
print("All 3 models trained. The optimal model has been exported to 'model.pkl'!")