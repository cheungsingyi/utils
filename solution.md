Understood. We'll skip formal validation and focus 100% on a "pure discovery" phase. The goal is to conduct a deep, comprehensive data exploration to find every possible signal and interaction, which we can then forge into a dynamic, multi-layered rule-based system.

Here’s a detailed implementation plan to exploit the most from our 6-month dataset.

---

## 1. Data Foundation: The "3-Month Analysis Cohort"

We cannot use all 6 months of data for a single analysis, as the performance window would be inconsistent (a 6-month-old customer has more time to perform than a 1-month-old one).

Your initial plan is the correct one, adapted for pure exploration:

* **Analysis Cohort:** Customers onboarded in **Months 1, 2, and 3**.
* **Performance Window:** We use data from **Months 4, 5, and 6** to calculate the 90-day performance for our cohort. (e.g., a customer from Month 1 is measured from Month 1 to Month 4).
* **Result:** This gives us a static, clean "master table" where each row is a customer from the cohort, containing all their onboarding form data and their known 90-day outcome metrics (e.g., revenue, products, transaction volume).

---

## 2. Target Definition: Creating a Composite "Potential Score"

Instead of a simple binary "high/low" flag, a comprehensive approach requires a more nuanced target. We will create a composite potential score for every customer in our Analysis Cohort.

**Implementation:**

* **Gather Outcomes:** For each customer, get their 90-day totals for:
    * `Revenue_90Day` (e.g., fees, interest)
    * `Product_Count_90Day` (how many distinct products they adopted)
    * `Activity_Volume_90Day` (e.g., total $ value of transactions)
* **Scale:** These metrics are on different scales (e.g., dollars vs. counts). We must scale them (e.g., using `MinMaxScaler` from `sklearn.preprocessing`) so they all range from 0 to 1.
* **Composite Score:** We create a weighted score. To start, we can weight them equally:
    > `Potential_Score = (0.33 * Scaled_Revenue) + (0.33 * Scaled_Product_Count) + (0.33 * Scaled_Activity)`
* **Create Tiers:** We can now segment our historical cohort by this score.
    * **Tier A (Platinum):** Top 10% (by `Potential_Score`)
    * **Tier B (Gold):** Next 15% (10%-25%)
    * **Tier C (Standard):** The rest

Now we have a rich target variable (`Tier`) to analyze against our form data.

---

## 3. Deep Exploration & Rule Discovery Techniques

This is where we find the rules. We will use three complementary techniques.

### Technique 1: 2D Interaction Heatmaps (Visual Rule Finding)

This goes beyond simple 1-variable analysis. We'll create 2D pivot tables to see how two form fields interact to predict potential.

**Implementation:**

* **Bin Numeric Data:** Convert `Self-Reported Revenue` and `Number of Employees` into bins (e.g., Low, Med, High, V-High).
* **Create 2D Heatmaps:** We will create a matrix, for example:
    * **Rows:** `Industry`
    * **Columns:** `Self-Reported Revenue (Binned)`
    * **Cell Value:** The average `Potential_Score` for that segment.
* **Rule Deduced:** By using conditional formatting (a heatmap), we can visually spot the "hot" red squares.
    > **Example Rule:** "We see a bright red spot for `Industry` = 'Manufacturing' and `Revenue_Bin` = 'V-High'. This is a high-potential segment."

### Technique 2: Decision Tree Rule Extraction (The "Dynamic" Engine)

This is the most powerful method for automatically finding dynamic, multi-step rules. We will use a shallow Decision Tree not for (ML) prediction, but as an EDA tool to find the most significant splits in our data.

**Implementation:**

* **Prepare Data:** Use the form data (e.g., `Industry`, `Revenue`, `Employees`) as our features (X). Use the `Tier` (A, B, C) as our target (y).
* **Fit Model:** Fit a `DecisionTreeClassifier` (from `sklearn.tree`) on this data.
* **CRITICAL:** We must constrain the tree to be simple and human-readable by setting `max_depth=3` or `max_depth=4`.
* **Export Rules:** We then use `sklearn.tree.export_text` to print the entire tree structure. This structure *is* our rule engine.
* **Example Text Output (Our Rules):**
    ```
    |--- Reported_Revenue <= 10.5M
    |   |--- Industry IN ['Retail', 'Hospitality']
    |   |   |--- class: Tier C
    |   |--- Industry IN ['Manufacturing', 'Software']
    |   |   |--- class: Tier B
    |--- Reported_Revenue > 10.5M
    |   |--- Products_of_Interest_Includes_FX = True
    |   |   |--- class: Tier A
    |   |--- Products_of_Interest_Includes_FX = False
    |   |   |--- Employees <= 50
    |   |   |   |--- class: Tier B
    |   |   |--- Employees > 50
    |   |   |   |--- class: Tier A
    ```
* **Rule Deduced:** We can now directly translate this. A client with >$10.5M revenue who also has >50 employees is Tier A, *even if they didn't ask for FX*. This is a dynamic, multi-step rule we would never find manually.

### Technique 3: Association Rule Mining (Finding "Hidden Profiles")

This "market basket analysis" technique finds what "sets" of form attributes are frequently associated with our Tier A customers.

**Implementation:**

* **Create "Baskets":** We create a dataset of all customers. Each customer is a "transaction," and their "items" are all their form attributes (e.g., `Industry=Manufacturing`, `Revenue=High`, `Employees=Mid`) plus their `Tier=A` flag.
* **Run Algorithm:** We use an algorithm like `Apriori` or `FP-Growth` to find frequent itemsets.
* **Find Rules:** We then filter for all rules where our `Tier=A` (or `Tier=B`) is the consequent (the item on the right side).
* **Example Rule Found:**
    > `{Industry='Software', Employees='>100'} => {Tier=A}`
* **Metrics:**
    * **Support:** 5% (5% of all our clients are 'Software', '>100 Employees', and 'Tier A').
    * **Confidence:** 65% (Of all clients who are 'Software' and '>100 Employees', 65% of them turned out to be Tier A).
* **Rule Deduced:** This gives us a new, high-confidence rule: `IF Industry='Software' AND Employees='>100'`, flag as High Potential.

---

## 4. Synthesizing into a "Dynamic Scorecard"

Finally, we combine all these findings into a comprehensive, points-based scoring system.

* **Heatmap Rules (Base Points):**
    * `IF Industry IN ['Manufacturing', 'Wholesale']: +10 points`
    * `IF Revenue_Bin = 'V-High': +15 points`
* **Decision Tree Rules (Interaction Points):**
    * `IF Revenue > $10.5M AND Products_of_Interest_Includes_FX = True: +25 points`
    * `IF Revenue > $10.5M AND Employees > 50: +20 points`
* **Association Rules (Hidden Profile Points):**
    * `IF Industry = 'Software' AND Employees > 100: +18 points`

An incoming customer is run through this scorecard. We can then set new, data-driven thresholds for routing:

* **Score > 30 (Tier 1):** Route to Senior RM
* **Score 15-30 (Tier 2):** Route to Junior RM
* **Score < 15 (Tier 3):** Route to automated onboarding
