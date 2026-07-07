import pandas as pd

# Load the csv file you just created
df = pd.read_csv('credit_record.csv')

print("🎉 EPIC 1 SUCCESSFUL!")
print("The computer has successfully read your SkillWallet dataset columns:")
print(df.columns.tolist())