import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load our data file
df = pd.read_csv('credit_record.csv')

print("📊 Epic 2: Starting Data Visualization...")

# 2. Create Chart 1: Income distribution check
plt.figure(figsize=(6, 4))
sns.boxplot(x='Approved', y='TOTAL_ANUAL_INCOME', data=df)
plt.title('Income Level vs Credit Approval')
plt.savefig('income_vs_approval.png') # Saves the chart to your folder
plt.close()

# 3. Create Chart 2: Count of approvals by gender
plt.figure(figsize=(6, 4))
sns.countplot(x='GENDER', hue='Approved', data=df)
plt.title('Approval Status by Gender')
plt.savefig('gender_vs_approval.png') # Saves the chart to your folder
plt.close()

print("🎉 EPIC 2 SUCCESSFUL!")
print("Two analysis charts ('income_vs_approval.png' and 'gender_vs_approval.png') have been saved to your folder!")