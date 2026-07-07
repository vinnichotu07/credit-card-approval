import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

print("⚙️ Epic 3: Starting Advanced Data Pre-processing...")

# 1. Load our raw data file
df = pd.read_csv('credit_record.csv')

# --- TOPIC A: DROP DUPLICATE FEATURES ---
# Removes any rows that were accidentally entered twice
df = df.drop_duplicates()
print(" -> Removed duplicate entries if any existed.")

# --- TOPIC B: HANDLING MISSING VALUES ---
# If any values are missing, this fills them with the most common value (mode)
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])
print(" -> Finished checking and fixing missing values.")

# --- TOPIC C: DATA CLEANING AND MERGING ---
# Strips empty white spaces out of text values to keep data uniform
text_columns = ['GENDER', 'OWN_CAR', 'OWN_REALSTATE', 'TYPE_OF_INCOME', 'EDUCATION', 'FAMILY_STATUS', 'TYPE_HOUSING']
for col in text_columns:
    df[col] = df[col].astype(str).str.strip()
print(" -> Data cleaned and verified.")

# --- TOPIC D: FEATURE ENGINEERING ---
# Creating a brand-new helper metric column: Total Family Income per Member
df['INCOME_PER_MEMBER'] = df['TOTAL_ANUAL_INCOME'] / df['FAMILY_MEMBERS']
print(" -> Created new engineered feature: 'INCOME_PER_MEMBER'")

# --- TOPIC E: HANDLING CATEGORICAL VALUES ---
# Converts structural text labels to machine-readable number values
encoders = {}
for col in text_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Save the final cleaned data matrix for Epic 4
df.to_csv('cleaned_credit_record.csv', index=False)

# Save converters so Epic 5 can map incoming form data accurately
with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print("\n🎉 EPIC 3 SUCCESSFUL!")
print("All 5 required sub-topics have been processed into 'cleaned_credit_record.csv'!")