# High-Value Customer Onboarding Scorecard Analysis

**Goal**: Identify high-potential corporate customers at the onboarding stage using limited KYC data (approx. 6 months history).
**Method**: Rule-based Scorecard derived from Weight of Evidence (WoE) and Information Value (IV).
**Language**: Python

---

## 1. Data Simulation (Mock Data)
Since we don't have the raw data loaded, we will simulate a dataset that reflects your scenario:
- **6 months of data** (Jan to June).
- **Fields**: `registered_capital`, `office_location`, `company_age`, `industry`.
- **Missing Values**: `office_location` will have ~50% missing.
- **Revenue**: Transaction fees/deposits for the first 3 months.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(123)

n_customers = 2000

# Generate Onboarding Dates (Jan 1 to June 30, 2025)
start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 6, 30)
days_range = (end_date - start_date).days
random_days = np.random.randint(0, days_range, n_customers)
onboarding_dates = [start_date + timedelta(days=int(d)) for d in random_days]

# Generate Features
df = pd.DataFrame({
    'customer_id': range(1, n_customers + 1),
    'onboarding_date': onboarding_dates,
    
    # Feature 1: Registered Capital (Log-normal distribution to simulate wealth gap)
    'registered_capital': np.round(np.random.lognormal(mean=12, sigma=2, size=n_customers), 0),
    
    # Feature 2: Office Location (Categorical with 50% Missing)
    # We use NaN to represent missing values
    'office_location': np.random.choice(
        ['CBD', 'Industrial', 'Residential', np.nan], 
        size=n_customers, 
        p=[0.1, 0.2, 0.2, 0.5]
    ),
    
    # Feature 3: Industry
    'industry': np.random.choice(
        ['Trading', 'Tech', 'F&B', 'Consulting'], 
        size=n_customers
    ),
    
    # Performance Data (Revenue for Month 1, 2, 3)
    'rev_m1': np.random.uniform(0, 1000, n_customers),
    'rev_m2': np.random.uniform(0, 1000, n_customers),
    'rev_m3': np.random.uniform(0, 1000, n_customers)
})

# Add some signal: Boost revenue for CBD and High Capital to make the rules work
mask_cbd = (df['office_location'] == 'CBD')
mask_high_cap = (df['registered_capital'] > 1000000)

df.loc[mask_cbd, 'rev_m1'] *= 5
df.loc[mask_high_cap, 'rev_m1'] *= 2

print(df.head())
```

---

## 2. Target Definition (Handling Time-on-Books Bias)
**Challenge**: Customers from January have 6 months of data, June customers have only 1 month.
**Solution**: 
1. Use a **Standardized Window** (First 3 Months Average).
2. Only use **Mature Customers** (Onboarded >= 3 months ago) for training.
3. Define "High Value" using **Cohort-based Percentiles** (Top 20% of *that month's* intake).

```python
# 1. Calculate Tenure (Months on Book) assuming current date is July 1st
current_date = datetime(2025, 7, 1)
df['months_on_book'] = (current_date - df['onboarding_date']).dt.days / 30
df['onboarding_month'] = df['onboarding_date'].dt.to_period('M')

# 2. Filter for Mature Customers (Training Set)
# We only learn rules from customers who have had enough time to prove themselves (> 3 months)
df_train = df[df['months_on_book'] >= 3].copy()

# 3. Calculate Performance Metric (F3M Average)
df_train['f3m_avg_revenue'] = (df_train['rev_m1'] + df_train['rev_m2'] + df_train['rev_m3']) / 3

# 4. Define Target: Top 20% per Cohort (Month)
# This removes seasonal bias (e.g., Feb might be low season)
df_train['p80_revenue'] = df_train.groupby('onboarding_month')['f3m_avg_revenue'].transform(lambda x: x.quantile(0.80))

# Create Binary Target: 1 = High Value, 0 = Normal
df_train['target'] = (df_train['f3m_avg_revenue'] >= df_train['p80_revenue']).astype(int)

print(df_train['target'].value_counts())
```

---

## 3. Data Cleaning & Feature Engineering
**Strategy**:
- **Missing Values**: Do NOT drop. Convert `NaN` to "Missing" category.
- **Binning**: Convert continuous variables (Capital) into categories (Bins).

```python
# 1. Handle Missing Values (Explicitly make them a category)
df_train['office_location'] = df_train['office_location'].fillna('Missing')

# 2. Binning Numerical Variables (Registered Capital)
# We use qcut (Quantile Cut) to create 3 buckets: Low, Mid, High
df_train['capital_bin'] = pd.qcut(
    df_train['registered_capital'], 
    q=[0, 0.33, 0.66, 1], 
    labels=['Low_Cap', 'Mid_Cap', 'High_Cap']
)

# Select only columns needed for analysis
model_data = df_train[['target', 'office_location', 'capital_bin', 'industry']].copy()
model_data['capital_bin'] = model_data['capital_bin'].astype(str)

print(model_data.head())
```

---

## 4. IV & WoE Calculation (The Core Logic)
We calculate **Information Value (IV)** to find the best features and **Weight of Evidence (WoE)** to assign scores.

**Formula**:
$$WoE = \ln(\frac{\% Good}{\% Bad})$$
$$IV = (\% Good - \% Bad) \times WoE$$

```python
def calculate_iv_woe(df, feature, target):
    """
    Calculates WoE and IV for a given feature.
    """
    lst = []
    for val in df[feature].unique():
        all_cnt = df[df[feature] == val].shape[0]
        good_cnt = df[(df[feature] == val) & (df[target] == 1)].shape[0]
        bad_cnt = df[(df[feature] == val) & (df[target] == 0)].shape[0]
        lst.append([val, all_cnt, good_cnt, bad_cnt])
        
    data = pd.DataFrame(lst, columns=['Value', 'Count', 'Good', 'Bad'])
    
    # Calculate Distributions
    total_good = data['Good'].sum()
    total_bad = data['Bad'].sum()
    
    data['Dist_Good'] = data['Good'] / total_good
    data['Dist_Bad'] = data['Bad'] / total_bad
    
    # Calculate WoE and IV
    epsilon = 1e-10
    data['WoE'] = np.log((data['Dist_Good'] + epsilon) / (data['Dist_Bad'] + epsilon))
    data['IV'] = (data['Dist_Good'] - data['Dist_Bad']) * data['WoE']
    
    data = data.sort_values(by='WoE', ascending=False)
    
    return data, data['IV'].sum()

# Run Analysis on our features
features_to_analyze = ['office_location', 'capital_bin', 'industry']
iv_results = {}

for feat in features_to_analyze:
    iv_table, total_iv = calculate_iv_woe(model_data, feat, 'target')
    iv_results[feat] = iv_table
    print(f"Feature: {feat} | Total IV: {total_iv:.4f}")
```

---

## 5. Scorecard Generation (The Deliverable)
We convert WoE into simple integer scores for the business team.
**Rule**: `Score = Round(WoE * 10)`

```python
def generate_scorecard_table(iv_table):
    scorecard = iv_table.copy()
    scorecard['Score'] = (scorecard['WoE'] * 10).round().astype(int)
    
    # Add Business Recommendation
    conditions = [
        (scorecard['Score'] > 10),
        (scorecard['Score'] < -5)
    ]
    choices = ['Strong Positive', 'Negative Signal']
    scorecard['Recommendation'] = np.select(conditions, choices, default='Neutral')
    
    return scorecard[['Value', 'Count', 'Dist_Good', 'WoE', 'Score', 'Recommendation']]

# 1. Scorecard for Office Location
print("\n=== Scorecard Rule: Office Location ===")
office_scorecard = generate_scorecard_table(iv_results['office_location'])
print(office_scorecard.to_string(index=False))

# 2. Scorecard for Registered Capital
print("\n=== Scorecard Rule: Registered Capital ===")
capital_scorecard = generate_scorecard_table(iv_results['capital_bin'])
print(capital_scorecard.to_string(index=False))
```

### Interpretation for Business
1.  **Look at the "Missing" row in Office Location**:
    - If the Score is negative (e.g., -5), it proves our hypothesis: "Customers who don't provide addresses are lower value."
    - **Action**: We don't drop them; we just give them a lower score.
2.  **Look at "CBD"**:
    - Likely has a high positive score.
3.  **Application**:
    - New Customer comes in:
        - Address: "Missing" (-5 points)
        - Capital: "High_Cap" (+10 points)
        - **Total Score**: 5 points.
    - Compare this total against your threshold (e.g., > 20 points = VIP).

---

## 6. Enhancement: Visualization (Optional)
Visualizing the WoE helps explain the "Why" to stakeholders.

```python
def plot_woe(iv_table, feature_name):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Value', y='WoE', data=iv_table, palette='viridis')
    plt.title(f'Weight of Evidence (WoE) by {feature_name}')
    plt.axhline(0, color='black', linestyle='--')
    plt.ylabel('WoE (Log Odds)')
    plt.show()

# Example Plot
# plot_woe(iv_results['office_location'], 'Office Location')
```

---

## 7. Next Steps
1.  **Run this Workflow**: Execute the Python code on your full dataset.
2.  **Filter Features**: Drop features with IV < 0.02.
3.  **Export Rules**: Save the final Score tables to Excel/SQL for the operations team.
