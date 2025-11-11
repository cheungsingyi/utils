# Updated Solution: Rule Derivation from 6-Month Data

The short timeframe means we must be very precise. We will use a 3-month cohort to build the rules and a hold-back validation set to prove they work.

**Data Split (The 6-Month Plan):**

* **Months 1 & 2 (Training Set):** We will use customers onboarded in the first two months.
* **Month 3 (Validation Set):** We will use customers onboarded in the third month to test our rules.
* **Months 4, 5, & 6 (Performance Window):** This data is used *only* to calculate the outcomes for our 3-month cohort.

Here is the step-by-step process to deduce the rules from this data.

---

## Step 1: Define the "90-Day High Potential" Target

For every customer in our **Months 1 & 2 Training Set**, we will calculate their performance in their first 90 days.

* A customer onboarded on Jan 15th will be measured on their activity from Jan 15th to Apr 15th. We have all this data.
* We can create a "90-Day Potential Score" for each customer:
    * **Revenue Component:** (90-Day Revenue Generated) \* (Weight A)
    * **Product Component:** (Number of Products Adopted) \* (Weight B)
    * **Activity Component:** (Volume of Transactions) \* (Weight C)
* We then find the **top 20%** of customers based on this score and flag them as `Is_High_Potential = 1`. This is our target.

---

## Step 2: Deduce Rules by Analyzing Segments

Now we can directly find the rules by comparing the onboarding form data (our "features") against the `Is_High_Potential` flag.

We will do this for two types of data:

### A. For Categorical Data (e.g., Industry, Products Interested In)

We will "group by" each field and calculate the "High Potential Rate" for each segment.

**Example Rule Derivation (Industry):**

| Industry (from form) | Total Customers | High Potential (Is\_High\_Potential = 1) | High Potential Rate | Rule? |
| :--- | :--- | :--- | :--- | :--- |
| **Manufacturing** | 50 | 20 | **40%** | **Yes (Strong Signal)** |
| Retail | 100 | 10 | 10% | No (Baseline) |
| Real Estate | 80 | 8 | 10% | No (Baseline) |
| **Wholesale Trade** | 45 | 15 | **33%** | **Yes (Strong Signal)** |
| *Overall Avg* | *275* | *53* | *19%* | *---* |

**Rule Deduced:** `IF Industry IN ['Manufacturing', 'Wholesale Trade']`... this is a high-potential segment.

### B. For Numeric Data (e.g., Self-Reported Revenue, Employees)

We will "bin" the data into quantiles (e.g., 5 groups, 20% each) and find the "sweet spot."

**Example Rule Derivation (Self-Reported Revenue):**

| Revenue Bin (from form) | Total Customers | High Potential (Is\_High\_Potential = 1) | High Potential Rate | Rule? |
| :--- | :--- | :--- | :--- | :--- |
| $0 - $1M | 60 | 3 | 5% | No |
| $1M - $5M | 70 | 10 | 14% | No |
| $5M - $10M | 65 | 11 | 17% | No |
| **$10M - $25M** | 50 | 14 | **28%** | **Yes** |
| **$25M+** | 30 | 15 | **50%** | **Yes (Strong Signal)** |

**Rule Deduced:** `IF Self-Reported Revenue > $10,000,000`... this is a high-potential segment.

---

## Step 3: Build the Triage Engine (From Rules)

We now combine these deduced rules into our engine.

**1. "Red Flag" Rules (Immediate Manual Route)**
These are for our strongest signals, often combinations.
* `IF Self-Reported Revenue > $25M`
* `IF Industry = 'Manufacturing' AND Self-Reported Revenue > $10M`
* `IF "Products of Interest" INCLUDES ['FX', 'Trade Finance']`

**2. Points-Based Scoring (Nuanced Triage)**
We can use the "lift" (how much better than average) to assign points.
* `Industry = 'Manufacturing'` (40% vs 19% avg. -> ~2x lift) -> **+10 points**
* `Revenue > $10M` (28% vs 19% avg. -> ~1.5x lift) -> **+5 points**
* `Industry = 'Retail'` (10% vs 19% avg. -> <1x lift) -> **0 points**

---

## Step 4: Validate the Rules

This is the most critical step.

1.  **Apply Rules to Validation Set:** We take our completed rule engine (built on Months 1-2) and run the **Month 3** customers through it.
2.  **Measure Performance:** The engine will flag, for example, 30 customers from Month 3 as "High Potential."
3.  **Check Reality:** We then wait and look at their actual 90-day performance (data from Months 4, 5, 6). Did the 30 customers we flagged *actually* perform better than the ones we didn't?
4.  **Iterate:** If our rules successfully predicted the top performers, they are ready for production. If not, we adjust the point thresholds and re-test.

This method uses our limited data to create rules that are directly tied to a measurable, short-term business outcome.
