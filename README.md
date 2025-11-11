# Project Documentation: High-Potential Customer Triage

**Project Goal:** To accelerate revenue generation by developing a data-driven, rule-based system that automatically triages new commercial clients, flagging high-potential customers for high-touch, manual processing by the sales team.

## Executive Summary

This project directly addresses a key business challenge:
> "How do we efficiently allocate our best sales resources to the new customers who will generate the most revenue?"

Our solution is to stop treating all new customers equally. We will implement a **"Fixed Benchmark" Triage Model** that uses our rich historical customer data to build a stable, data-driven "ruler" for measuring potential.

Using this benchmark, we will analyze the onboarding form data from our recent 6-month cohort to find the *"tells"*—the key data points and combinations that predict a high-value client before they even complete onboarding.

The final deliverable is a **Dynamic Triage Scorecard**. This is a simple, points-based engine that runs in real-time, instantly scoring every new applicant and routing them:

* **Tier A (High Potential):** Immediate, high-priority route to a Senior Relationship Manager (RM).
* **Tier B (Medium Potential):** Standard route to a Junior RM or assisted-digital channel.
* **Tier C (Standard Potential):** Default to a fully-automated digital onboarding path.

### Business Impact:

* **Increased Sales Conversion:** Our best RMs spend their time on pre-qualified, high-value leads.
* **Higher Customer Lifetime Value:** High-potential clients receive the high-touch service they require from Day 1, leading to deeper product penetration.
* **Maximum Sales Efficiency:** We reduce time spent on low-potential clients, automating their journey and lowering our cost-to-serve.
* **Stable Business Metrics:** We create a fixed, objective definition of a "Tier A" client, allowing us to track onboarding quality month-over-month.

---

## Core Methodology: The "Fixed Benchmark" Model

This is the optimal solution for this project.

* **The Challenge:** If we only use our 6-month "analysis cohort" to define "high potential" (e.g., "the top 10% of that group"), our definition is relative. The "best" customer in a bad month would look the same as the "best" customer in a great month. This creates an unstable, "floating" target that is useless for future prediction.
* **The Solution:** We will use our large, mature **"Reference Population"** (all existing customers onboarded 6-24 months ago) to create a fixed, permanent benchmark.

We use this large, stable dataset one time to answer: *"What does a good 90-day performance actually look like, across our entire history?"*

This gives us **absolute thresholds** for "Tier A" and "Tier B" potential. These thresholds are now fixed. We then use this benchmark to label our 6-month analysis cohort, giving us a stable, reliable target for our rule discovery.

---

## Phase 1: Benchmark Creation (Building the "Ruler")

**Objective:** To create and save a "Benchmark Model" (a scaler and two threshold numbers) that defines "potential" in a stable, absolute way.

**Data:** `Reference_Population.csv` (Your large, mature dataset of all customers, with their 90-day performance metrics).

**Implementation Steps:**

1.  **Load Reference Population:** Load the historical data for all mature customers.
2.  **Define Outcome Metrics:** We will define "potential" as a composite of three 90-day metrics:
    * `Revenue_90Day`: Total revenue (fees, interest, etc.).
    * `Product_Count_90Day`: Number of distinct products adopted.
    * `Activity_Volume_90Day`: Total value of transactions.
3.  **Fit "Master Scaler":** To make these metrics comparable (e.g., dollars vs. counts), we fit a `MinMaxScaler` on these three columns. This scaler is fit *only* on this dataset, one time.
4.  **Save the Scaler:** We save this `master_scaler.pkl` object. It now "knows" the true historical minimum and maximum for all our metrics.
5.  **Calculate Potential Score:** We use the scaler to transform the data and create a `Potential_Score` for all historical customers (e.g., `0.33 * Scaled_Revenue + ...`).
6.  **Find Absolute Thresholds:** We find the 90th and 75th percentile of this stable score.
    * `TIER_A_THRESHOLD = Potential_Score.quantile(0.90)` (e.g., 0.823)
    * `TIER_B_THRESHOLD = Potential_Score.quantile(0.75)` (e.g., 0.651)
7.  **Store Thresholds:** These two numbers (e.g., 0.823 and 0.651) are now our permanent, fixed definitions of Tier A and Tier B.

**Deliverable:**
* `master_scaler.pkl`
* `benchmark_config.json` (containing the two threshold values)

---

## Phase 2: Rule Discovery (The "Analysis")

**Objective:** To find which onboarding form inputs (e.g., Industry, Revenue) from our recent cohort successfully predict the Tiers defined in Phase 1.

**Data:** `Analysis_Cohort.csv` (Your 6-month dataset containing Form Data and 90-Day Outcome Metrics).

**Implementation Steps:**

1.  **Load Analysis Cohort:** Load the data for the new customers (Months 1-3).
2.  **Load Benchmark Model:** Load `master_scaler.pkl` and the two threshold values.
3.  **Apply Benchmark to Cohort:**
    * Use the loaded scaler to **transform** (not fit) this cohort's 90-day outcome metrics.
    * Calculate their `Potential_Score` using the same formula.
    * Assign Tiers: `IF Potential_Score > 0.823 -> Tier = 'A'`, etc.
4.  **This `Tier` column is now our reliable target (y) for analysis.**
5.  **Run Rule-Discovery EDA:** We now search for predictors (X) for this target.
    * **Method 1: 2D Interaction Heatmaps:** We create pivot tables (e.g., `Industry` vs. `Revenue_Bin`) to visually spot "hot" segments that have a high average `Potential_Score`. This finds simple, strong rules.
    * **Method 2: Decision Tree Rule Extraction:** This is our "dynamic" rule finder. We fit a shallow `DecisionTreeClassifier` (e.g., `max_depth=3`) to predict `Tier` using the form data. We then export the text of this tree. The tree's logic is our set of dynamic, multi-step rules (e.g., `IF Revenue > 10M AND Employees > 50 -> Tier A`).

**Deliverable:** A list of data-driven rules (e.g., `IF Industry = 'Manufacturing' -> +10 points`) that will form our scorecard.

---

## Phase 3: The Triage Scorecard & Deployment

**Objective:** To synthesize our rules into a simple, deployable points-based system.

**Implementation:**

1.  **Synthesize Rules:** We review the outputs from our heatmaps and decision tree and assign points to each rule.
    * **Example: Base Rule (from heatmap)**
        * `IF Industry IN ['Manufacturing', 'Wholesale'] -> +10 points`
        * `IF Self_Reported_Revenue > 10M -> +15 points`
    * **Example: Dynamic Rule (from tree)**
        * `IF Self_Reported_Revenue > 10M AND Number_of_Employees > 50 -> +20 points` (This is an additional bonus)
2.  **Create Triage Function:** We create a simple Python function (`triage_new_customer`) that takes a new customer's form data (as a JSON/dict), applies the scorecard rules, and sums the points.
3.  **Set Routing Thresholds:** We define the final routing logic based on the total score.
    * `IF Total_Score > 30 -> Route to Senior RM (Tier A)`
    * `IF Total_Score > 15 -> Route to Junior RM (Tier B)`
    * `ELSE -> Route to Automated (Tier C)`

This function is now ready to be integrated into your onboarding application.

---

## Alternative Solutions (For Reference)

### Relative Cohort Model (Not Recommended):
* **Method:** Use only the 6-month cohort. Find the top 10% of that group and call them "Tier A."
* **Pro:** Simpler, requires no historical data.
* **Con (Critical):** The definition of "Tier A" is unstable and relative. It doesn't tell you if a group is *actually* good, just *better than the rest*. This model fails when market conditions change (e.g., a "bad" month has no true Tier A customers, but the model is forced to find some).

### Full Machine Learning Model (Future Step):
* **Method:** Instead of a scorecard, we would deploy the `DecisionTreeClassifier` (or a more complex model like XGBoost) as a predictive API.
* **Pro:** More accurate, can find much more complex patterns.
* **Con:** Less transparent ("black box"), harder for the business to understand, and requires more infrastructure to deploy and monitor.
* **Recommendation:** This is the logical Phase 2 of this project. Our current rule-based system provides immediate value and creates the perfect, well-defined target (Tier) needed to train this future ML model.

---

## Next Steps

1.  **Validation:** Before deployment, we will backtest this scorecard. We will run it on a "hold-out" set (e.g., customers from Month 4) and see if our rules would have correctly predicted their known 90-day performance.
2.  **Deployment:** Integrate the `triage_new_customer` function into the live onboarding workflow.
3.  **Monitor & Iterate:** We will track the 90-day performance of our new, routed cohorts (Tier A, B, C) to prove the system is working and generating a lift in revenue.
