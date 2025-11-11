High-Potential Customer Triage - Project CodeThis file contains all the Python code for the customer triage project, organized by phase.1. 1_create_benchmark_model.pyThis script (Phase 1) uses the large, mature 'Reference Population' dataset to create and save the stable 'Benchmark Model' (master_scaler.pkl and benchmark_config.json).```python"""Phase 1: Benchmark Creation (Building the "Ruler")This script uses the large, mature 'Reference Population' dataset.Its purpose is to define "potential" in a stable, absolute way.It generates two critical assets:'master_scaler.pkl': A scikit-learn scaler object.'benchmark_config.json': A file with the absolute score thresholds."""import pandas as pdfrom sklearn.preprocessing import MinMaxScalerimport pickleimport jsonimport numpy as np # Import numpyimport os # Import os to check for data directory--- Configuration ---DATA_DIR = 'data'REFERENCE_DATA_FILE = os.path.join(DATA_DIR, 'reference_population.csv') # Your large historical datasetMETRICS_TO_SCALE = ['Revenue_90Day', 'Product_Count_90Day', 'Activity_Volume_90Day']SCORE_WEIGHTS = {'Revenue_90Day': 0.4,'Product_Count_90Day': 0.3,'Activity_Volume_90Day': 0.3}TIER_A_PERCENTILE = 0.90  # Top 10%TIER_B_PERCENTILE = 0.75  # Top 25%--- Mock Data Generation (Remove when using real data) ---def get_mock_reference_data():"""Generates a large mock dataset."""print("Loading mock reference data...")data = {'CustomerID': range(1, 5001),'Revenue_90Day': np.random.gamma(2, 10000, 5000),'Product_Count_90Day': np.random.randint(1, 6, 5000),'Activity_Volume_90Day': np.random.gamma(3, 50000, 5000)}df = pd.DataFrame(data)# Add some high-value outliersoutlier_indices = np.random.choice(df.index, 500, replace=False)df.loc[outlier_indices, 'Revenue_90Day'] *= 5df.loc[outlier_indices, 'Activity_Volume_90Day'] *= 3df.loc[outlier_indices, 'Product_Count_90Day'] = np.random.randint(4, 8, 500)# Ensure data directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Save mock data to be read by the script
df.to_csv(REFERENCE_DATA_FILE, index=False)
print(f"Mock reference data saved to '{REFERENCE_DATA_FILE}'")
return df
--- End of Mock Data ---def create_benchmark():"""Main function to create and save the benchmark model."""print("--- Phase 1: Creating Benchmark Model ---")# 1. Load Reference Population
try:
    df = pd.read_csv(REFERENCE_DATA_FILE)
    print(f"Successfully loaded '{REFERENCE_DATA_FILE}'")
except FileNotFoundError:
    print(f"Warning: '{REFERENCE_DATA_FILE}' not found. Generating mock data.")
    df = get_mock_reference_data() # This will generate and save the file

if df[METRICS_TO_SCALE].isnull().any().any():
    print("Warning: Data contains null values. Dropping rows with nulls.")
    df = df.dropna(subset=METRICS_TO_SCALE)

print(f"Loaded {len(df)} mature customer records.")

# 2. Fit "Master Scaler"
print(f"Fitting Master Scaler on: {METRICS_TO_SCALE}")
scaler = MinMaxScaler()
scaler.fit(df[METRICS_TO_SCALE])

# 3. Save the Scaler
scaler_filename = 'master_scaler.pkl'
with open(scaler_filename, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Successfully saved scaler to '{scaler_filename}'")

# 4. Calculate Potential Score
print("Calculating 'Potential_Score' for all historical customers...")
scaled_data = scaler.transform(df[METRICS_TO_SCALE])

# Apply weights
df['Potential_Score'] = (
    scaled_data[:, 0] * SCORE_WEIGHTS['Revenue_90Day'] +
    scaled_data[:, 1] * SCORE_WEIGHTS['Product_Count_90Day'] +
    scaled_data[:, 2] * SCORE_WEIGHTS['Activity_Volume_90Day']
)

# 5. Find Absolute Thresholds
tier_a_threshold = df['Potential_Score'].quantile(TIER_A_PERCENTILE)
tier_b_threshold = df['Potential_Score'].quantile(TIER_B_PERCENTILE)

print("\n--- Benchmark Creation Complete ---")
print(f"Tier A Threshold ({TIER_A_PERCENTILE*100}th percentile): {tier_a_threshold:.4f}")
print(f"Tier B Threshold ({TIER_B_PERCENTILE*100}th percentile): {tier_b_threshold:.4f}")

# 6. Store Thresholds
benchmark_config = {
    'TIER_A_THRESHOLD': tier_a_threshold,
    'TIER_B_THRESHOLD': tier_b_threshold,
    'metrics_scaled': METRICS_TO_SCALE,
    'score_weights': SCORE_WEIGHTS
}

config_filename = 'benchmark_config.json'
with open(config_filename, 'w') as f:
    json.dump(benchmark_config, f, indent=4)
print(f"Successfully saved config to '{config_filename}'")
if name == "main":create_benchmark()```2. 2_discover_triage_rules.pyThis script (Phase 2) uses the 'Analysis Cohort' (6-month data) and applies the 'Benchmark Model' to it. It then runs exploratory data analysis (EDA) to find the predictive rules for our scorecard.```python"""Phase 2: Rule Discovery (The "Analysis")This script uses the 'Analysis Cohort' (e.g., the 6-month dataset).It loads the Benchmark Model from Phase 1 to label this cohort.Then, it runs EDA techniques (Heatmaps, Decision Trees) to findthe form inputs (X) that predict the stable Tiers (y).The output of this script is NOT a program, but the human-readableinsights and rules that we will use to build the Phase 3 Scorecard."""import pandas as pdimport pickleimport jsonimport seaborn as snsimport matplotlib.pyplot as pltfrom sklearn.tree import DecisionTreeClassifier, export_textfrom sklearn.preprocessing import OneHotEncoderfrom sklearn.compose import ColumnTransformerimport numpy as np # Import numpyimport os # Import os--- Configuration ---DATA_DIR = 'data'ANALYSIS_DATA_FILE = os.path.join(DATA_DIR, 'analysis_cohort.csv') # Your 6-month cohort dataBENCHMARK_SCALER_FILE = 'master_scaler.pkl'BENCHMARK_CONFIG_FILE = 'benchmark_config.json'Form fields to use as features (X)FORM_FEATURES = ['Self_Reported_Revenue', 'Number_of_Employees', 'Industry', 'Years_in_Business']CATEGORICAL_FEATURES = ['Industry']NUMERIC_FEATURES = ['Self_Reported_Revenue', 'Number_of_Employees', 'Years_in_Business']--- Mock Data Generation (Remove when using real data) ---def get_mock_analysis_data():"""Generates a 6-month cohort mock dataset."""print("Loading mock analysis cohort data...")N = 1000industries = ['Manufacturing', 'Retail', 'Software', 'Hospitality', 'Real Estate']data = {'CustomerID': range(5001, 5001 + N),# Features (X)'Self_Reported_Revenue': np.random.gamma(2, 5000000, N).clip(100000, 50000000),'Number_of_Employees': np.random.randint(1, 500, N),'Industry': np.random.choice(industries, N),'Years_in_Business': np.random.randint(0, 30, N),# Outcomes (y_raw)'Revenue_90Day': np.random.gamma(2, 8000, N),'Product_Count_90Day': np.random.randint(1, 4, N),'Activity_Volume_90Day': np.random.gamma(3, 40000, N)}df = pd.DataFrame(data)# Manually create correlation: 'Software' and 'Manufacturing' with high revenue = better outcomes
high_potential_mask = (df['Industry'].isin(['Software', 'Manufacturing'])) & (df['Self_Reported_Revenue'] > 20000000)
df.loc[high_potential_mask, 'Revenue_90Day'] *= 5
df.loc[high_potential_mask, 'Product_Count_90Day'] = np.random.randint(3, 6, high_potential_mask.sum())

# Ensure data directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    
df.to_csv(ANALYSIS_DATA_FILE, index=False)
print(f"Mock analysis data saved to '{ANALYSIS_DATA_FILE}'")
return df
--- End of Mock Data ---def analyze_cohort():"""Main function to load cohort, apply benchmark, and discover rules."""print("--- Phase 2: Discovering Triage Rules ---")# 1. Load Analysis Cohort
try:
    df = pd.read_csv(ANALYSIS_DATA_FILE)
    print(f"Successfully loaded '{ANALYSIS_DATA_FILE}'")
except FileNotFoundError:
    print(f"Warning: '{ANALYSIS_DATA_FILE}' not found. Generating mock data.")
    df = get_mock_analysis_data()

print(f"Loaded {len(df)} new customer records for analysis.")

# 2. Load Benchmark Model
try:
    scaler = pickle.load(open(BENCHMARK_SCALER_FILE, 'rb'))
    with open(BENCHMARK_CONFIG_FILE, 'r') as f:
        config = json.load(f)
    
    TIER_A_THRESHOLD = config['TIER_A_THRESHOLD']
    TIER_B_THRESHOLD = config['TIER_B_THRESHOLD']
    METRICS_TO_SCALE = config['metrics_scaled']
    SCORE_WEIGHTS = config['score_weights']
    print(f"Successfully loaded benchmark model from '{BENCHMARK_SCALER_FILE}' and '{BENCHMARK_CONFIG_FILE}'")
except FileNotFoundError:
    print("Error: Benchmark model files not found. Please run '1_create_benchmark_model.py' first.")
    return

# 3. Apply Benchmark to Cohort
print("Applying benchmark to analysis cohort...")

# Ensure cohort has the required outcome metrics
if not all(col in df.columns for col in METRICS_TO_SCALE):
    print(f"Error: Cohort data is missing one of the required outcome metrics: {METRICS_TO_SCALE}")
    return
    
# Drop rows where outcome metrics are missing
df_with_outcomes = df.dropna(subset=METRICS_TO_SCALE)
if len(df_with_outcomes) == 0:
    print("Error: No data left after dropping rows with missing outcomes. Cannot proceed.")
    return

scaled_data = scaler.transform(df_with_outcomes[METRICS_TO_SCALE])

df_outcomes = pd.DataFrame(scaled_data, columns=METRICS_TO_SCALE, index=df_with_outcomes.index)

df_outcomes['Potential_Score'] = (
    df_outcomes[METRICS_TO_SCALE[0]] * SCORE_WEIGHTS['Revenue_90Day'] +
    df_outcomes[METRICS_TO_SCALE[1]] * SCORE_WEIGHTS['Product_Count_90Day'] +
    df_outcomes[METRICS_TO_SCALE[2]] * SCORE_WEIGHTS['Activity_Volume_90Day']
)

# Assign stable tiers (Our Target, y)
df_outcomes['Tier'] = 'C'
df_outcomes.loc[df_outcomes['Potential_Score'] > TIER_B_THRESHOLD, 'Tier'] = 'B'
df_outcomes.loc[df_outcomes['Potential_Score'] > TIER_A_THRESHOLD, 'Tier'] = 'A'

# Join Tiers back to the main dataframe
df = df.join(df_outcomes[['Potential_Score', 'Tier']])
print("Tiers assigned to cohort:")
print(df['Tier'].value_counts(normalize=True).sort_index())

# 4. Run Rule-Discovery EDA
print("\n--- EDA 1: 2D Interaction Heatmap ---")

# Bin a numeric variable for the heatmap
try:
    df['Revenue_Bin'] = pd.qcut(df['Self_Reported_Revenue'], 4, labels=['Low', 'Med', 'High', 'V-High'])
    heatmap_data = df.groupby(['Industry', 'Revenue_Bin'])['Potential_Score'].mean().unstack()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.3f')
    plt.title('Heatmap: Avg. Potential Score by Industry & Reported Revenue')
    plt.ylabel('Industry')
    plt.xlabel('Reported Revenue Bin')
    plt.savefig('heatmap_potential_score.png')
    print("Saved 'heatmap_potential_score.png'")
    # plt.show() # Uncomment to display plot
    
    print("\n** Heatmap Insights: Look for 'hot' (high-value) squares in the heatmap. **")
    print("Example Rule: 'Industry=Software' + 'Revenue_Bin=V-High' has a high avg score.")
    
except Exception as e:
    print(f"Could not generate heatmap: {e}")

print("\n--- EDA 2: Decision Tree Rule Extraction ---")

# Prep data for the tree
df_tree = df.dropna(subset=FORM_FEATURES + ['Tier'])
X = df_tree[FORM_FEATURES]
y = df_tree['Tier']

if len(X) == 0:
    print("No data available for Decision Tree analysis after dropping nulls.")
    return

# Create a preprocessor to handle mixed types (numeric and categorical)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', NUMERIC_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
    ],
    remainder='drop'
)

X_processed = preprocessor.fit_transform(X)

# Get feature names after one-hot encoding
try:
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(CATEGORICAL_FEATURES)
except AttributeError:
    # Fallback for older scikit-learn versions
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names(CATEGORICAL_FEATURES)
    
feature_names = NUMERIC_FEATURES + list(cat_feature_names)

# Fit the Decision Tree
# We use class_weight='balanced' to handle imbalanced Tiers
tree_model = DecisionTreeClassifier(max_depth=3, class_weight='balanced', min_samples_leaf=20)
tree_model.fit(X_processed, y)

# Export the rules as text
rules = export_text(tree_model, feature_names=feature_names)

print("\n** Decision Tree Rules (Copy these to build your scorecard) **")
print("===============================================================")
print(rules)
print("===============================================================")
print("\n** How to read the tree: **")
print("This tree shows the *most important* splits that predict a Tier.")
print("Example Rule: 'Self_Reported_Revenue <= 10.5M' and 'Industry_Software = 1.0' might lead to Tier B.")
if name == "main":analyze_cohort()```3. 3_triage_new_customer.pyThis script (Phase 3) contains the final, deployable function triage_new_customer. This function takes a new customer's form data, applies our new scorecard rules, and returns a routing decision. This is the script you integrate into your production application.```python"""Phase 3: The Triage Scorecard & DeploymentThis script contains the FINAL, DEPLOYABLE function.This function, 'triage_new_customer', takes the form datafrom a brand new customer application.It runs the rules we discovered in Phase 2 (which arenow hard-coded into the 'scorecard_rules') and returnsa final routing decision.This is the file you would integrate with your web application."""import jsondef triage_new_customer(form_data):"""Applies the data-driven Triage Scorecard to a new customer.Args:
    form_data (dict): A dictionary containing the new customer's
                      data from the onboarding form.
                      
Returns:
    dict: A dictionary containing the final score and routing decision.
"""

# --- The Triage Scorecard ---
# These rules are synthesized from the EDA in Phase 2.
# YOU MUST REPLACE THESE WITH THE RULES YOU DISCOVERED.
scorecard_rules = [
    # --- Rule Set 1: Base Points (from Heatmaps) ---
    {
        'name': 'High-Revenue Industry',
        'points': 10,
        'condition': lambda d: d.get('Industry') in ['Manufacturing', 'Software', 'Wholesale Trade']
    },
    {
        'name': 'Very High Reported Revenue',
        'points': 15,
        'condition': lambda d: d.get('Self_Reported_Revenue', 0) > 20000000
    },
    {
        'name': 'Medium-High Reported Revenue',
        'points': 10,
        'condition': lambda d: 10000000 < d.get('Self_Reported_Revenue', 0) <= 20000000
    },
    {
        'name': 'Established Business',
        'points': 5,
        'condition': lambda d: d.get('Years_in_Business', 0) > 10
    },
    
    # --- Rule Set 2: Dynamic Interaction Points (from Decision Tree) ---
    {
        'name': 'Interaction: High Revenue AND High Employees',
        'points': 20, # This is an *additional* bonus
        'condition': lambda d: d.get('Self_Reported_Revenue', 0) > 15000000 and d.get('Number_of_Employees', 0) > 50
    },
    {
        'name': 'Interaction: Software Startup',
        'points': 10,
        'condition': lambda d: d.get('Industry') == 'Software' and d.get('Years_in_Business', 0) < 3
    },
    
    # --- Rule Set 3: Risk/Low Potential (Negative Points) ---
    {
        'name': 'Low-Revenue Industry',
        'points': -5,
        'condition': lambda d: d.get('Industry') in ['Hospitality', 'Retail']
    }
]

# --- Routing Thresholds ---
ROUTING_THRESHOLDS = {
    'Tier A (High Potential)': 30,  # Route to Senior RM
    'Tier B (Medium Potential)': 15  # Route to Junior RM
}

# --- Scoring Logic ---
total_score = 0
rules_triggered = []

for rule in scorecard_rules:
    try:
        if rule['condition'](form_data):
            total_score += rule['points']
            rules_triggered.append(f"{rule['name']} (+{rule['points']} pts)")
    except TypeError:
        # This handles missing data, e.g., d.get('Revenue', 0) > 10M
        # The '0' default prevents a crash if 'Revenue' is missing.
        print(f"Warning: Skipping rule '{rule['name']}' due to missing/invalid data.")
        pass
        
# --- Final Routing Decision ---
routing_decision = 'Tier C (Standard Potential)' # Default
if total_score >= ROUTING_THRESHOLDS['Tier A (High Potential)']:
    routing_decision = 'Tier A (High Potential)'
elif total_score >= ROUTING_THRESHOLDS['Tier B (Medium Potential)']:
    routing_decision = 'Tier B (Medium Potential)'
    
return {
    'final_score': total_score,
    'routing_decision': routing_decision,
    'rules_triggered': rules_triggered
}
--- Example Usage (How you would run this in production) ---if name == "main":print("--- Running Triage Scorecard on New Customers ---")

# Example 1: A high-potential client
client_a = {
    "CustomerID": 6001,
    "Self_Reported_Revenue": 25000000,
    "Number_of_Employees": 75,
    "Industry": "Manufacturing",
    "Years_in_Business": 12
}

# Example 2: A medium-potential client
client_b = {
    "CustomerID": 6002,
    "Self_Reported_Revenue": 8000000,
    "Number_of_Employees": 20,
    "Industry": "Software",
    "Years_in_Business": 2
}

# Example 3: A standard-potential client
client_c = {
    "CustomerID": 6003,
    "Self_Reported_Revenue": 1000000,
    "Number_of_Employees": 5,
    "Industry": "Retail",
    "Years_in_Business": 8
}

for client in [client_a, client_b, client_c]:
    result = triage_new_customer(client)
    print(f"\nClient {client['CustomerID']} ({client['Industry']}):")
    print(json.dumps(result, indent=2))
```4. requirements.txtThis file lists the necessary Python packages to run the scripts.```textpandasscikit-learnseabornmatplotlibnumpy```
