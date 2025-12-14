# Credit Scoring Business Understanding

## Part 1: Fundamentals of Credit Risk and Regulation

### 1. The Concept of Credit Risk

Credit risk is defined as the potential that a borrower, counterparty, issuer, or obligor will fail to perform a financial or contractual obligation in accordance with the agreed-upon terms, resulting in an enforceable loss to the creditor or risk-bearing institution. In simple terms, it is the risk that a lender will not be paid back (principal and interest).

Credit Risk Management is the process banks and financial institutions use to maximize their risk-adjusted rate of return by keeping credit exposure within acceptable limits.

The risk is quantified by models that estimate three key parameters:

- **Probability of Default (PD)**: The likelihood that a borrower will default over a specific time horizon (e.g., 1 year).
- **Loss Given Default (LGD)**: The fraction of the exposure that a bank expects to lose if a default occurs.
- **Exposure at Default (EAD)**: The outstanding amount a bank is owed at the time of default.

### 2. The Basel II Capital Accord

The Basel Accords are a set of international banking regulations developed by the Basel Committee on Banking Supervision (BCBS) that set minimum capital requirements for banks. Basel II, introduced in 2004, transformed how credit risk is managed and regulated, making it a foundational component of statutory prudential frameworks.

Basel II is structured around Three Pillars:

| Pillar | Focus | Relevance to Credit Scoring |
|--------|-------|-----------------------------|
| Pillar 1: Minimum Capital Requirements | Defines the rules for calculating the minimum regulatory capital a bank must hold to cover credit, operational, and market risks. | Banks can use one of two main approaches for credit risk: the Standardized Approach or the Internal Ratings-Based (IRB) Approach. The IRB approach requires banks to develop sophisticated internal models (credit scoring models) to estimate PD, LGD, and EAD. |
| Pillar 2: Supervisory Review Process | Requires regulators to review a bank's internal capital adequacy assessment and risk management processes. | This pillar demands a robust Model Governance Framework. Regulators scrutinize the quality, validation, and management of the internal credit scoring models used by the bank. |
| Pillar 3: Market Discipline | Requires banks to disclose key information about their risk profile, risk management, and capital adequacy to the public. | This promotes transparency and forces banks to demonstrate sound risk management practices, including those embedded in their scoring models. |

## Part 2: Credit Scoring Business Understanding

### Credit Scoring Business Understanding

#### How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Accord (and subsequent Basel III) mandates that banks adopt a rigorous framework for measuring and managing risk, often through the Internal Ratings-Based (IRB) approach which relies on internal credit scoring models to calculate regulatory capital.

This emphasis directly influences model requirements:

- **Model Governance and Auditability**: Under Pillar 2, regulators demand an effective model governance framework. This requires that the credit model is not a "black box" but is interpretable and well-documented so that auditors and supervisors can understand its logic, validate its accuracy, and ensure it complies with regulatory standards.
- **Consumer Protection**: Decisions based on credit scoring must be explainable and transparent to the consumer. If a loan application is rejected based on a model's score, the financial institution must be able to clearly articulate the specific factors (features) that led to the low score. An interpretable model is essential for providing these "reason codes".
- **Mitigation of Bias and Risk**: Opaque algorithms can raise concerns about fairness, discrimination, and the potential to perpetuate historical biases. Regulatory guidance stresses that institutions must be able to explain, understand, and justify the decisions made by their scoring methods to mitigate these risks.

#### Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

Credit scoring is fundamentally a statistical method used to predict the Probability of Default (PD) or delinquency. The success of the model depends on having a clear binary target variable ($Y_i$) that defines what a "bad" borrower is (i.e., default).

**Why a Proxy is Necessary**:

- In many real-world scenarios, particularly for new loan products, thin-file customers, or markets with weak credit bureau reporting, a direct and universally accepted "default" label (like 90 days past due or bankruptcy) may not be consistently available or sufficiently observed in the historical data.
- A proxy variable (e.g., maximum utilization, persistent minimum payments, or account closure) is created to approximate the unobserved or rare event of true default, allowing the development of a predictive model.

**Potential Business Risks of Using a Proxy**:

- **Inaccurate Risk Measurement**: The most significant risk is that the model is predicting the proxy, not the true credit risk. If the proxy variable is a poor representation of actual default behavior, the resulting PD estimates will be skewed, leading to loan losses.
- **Mispricing of Risk**: An inaccurate model will either approve risky customers (leading to losses) or deny credit to creditworthy customers (leading to lost revenue).
- **Model Risk**: Regulatory guidance on model risk management requires that models are conceptually sound and fit for purpose. Using a conceptually weak or poorly correlated proxy introduces high model risk that could fail internal validation or external supervisory review.

#### What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

The choice between a simple and a complex model is a critical trade-off between Performance and Interpretability/Compliance:

| Model Type | Advantage in Regulated Context | Key Trade-off / Risk |
|------------|--------------------------------|----------------------|
| Simple (e.g., Logistic Regression) | High Interpretability: The model logic (Weights of Evidence - WoE) is linear, transparent, and easy to explain to regulators, auditors, and customers. This simplicity is often preferred for scorecards in highly regulated Pillar 1 calculations. | Lower Performance: It may miss complex, non-linear relationships in the data, resulting in lower predictive power (AUC) compared to sophisticated models. |
| Complex (e.g., Gradient Boosting) | High Performance: It can achieve superior accuracy by capturing complex relationships, potentially reducing actual credit losses. | Low Interpretability (Opaqueness): The model's decision-making process is complex and non-linear, making it a "black box" that is difficult to explain and justify. This requires significant effort and cost (e.g., using post-modelling interpretability techniques like LIME) to satisfy regulatory requirements for transparency and explanation. |

In a regulated environment, Interpretability and Compliance often outweigh marginal performance gains, making simple models a common industry choice for core risk decisions, while complex models are sometimes relegated to challenger or screening roles where regulatory scrutiny is less intense.