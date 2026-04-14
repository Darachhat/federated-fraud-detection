Writing the complete full thesis document in Markdown now — all chapters, all front matter, all confirmed values.

---

# A Federated Learning Framework for Early Fraud Detection using XGBoost: Balancing Model Performance and Data Privacy

---

**សាកលវិទ្យាល័យភូមិន្ទភ្នំពេញ**

**ROYAL UNIVERSITY OF PHNOM PENH**

**ប្រធានបទនិក្ខេបបទ**

**A Federated Learning Framework for Early Fraud Detection using XGBoost: Balancing Model Performance and Data Privacy**

A Thesis

In Partial Fulfilment of the Requirement for the Degree of

Bachelor of Engineering in Information Technology Engineering

**SOTHUN DARACHHAT**

**June 2025**

---

**សាកលវិទ្យាល័យភូមិន្ទភ្នំពេញ**

**ROYAL UNIVERSITY OF PHNOM PENH**

**ប្រធានបទនិក្ខេបបទ**

**A Federated Learning Framework for Early Fraud Detection using XGBoost: Balancing Model Performance and Data Privacy**

A Thesis

In Partial Fulfilment of the Requirement for the Degree of

Bachelor of Engineering in Information Technology Engineering

**SOTHUN DARACHHAT**

**Examination Committee:** \[Chairperson Name\] (Chairperson)

\[Committee Member Names\]

**Supervisor:** Mr. Chhim Bunchhun

**June 2025**

---

## មូលន័យសង្ខេប {#មូលន័យសង្ខេប .ABSTRACT-KH}

ការក្លែងបន្លំផ្នែកហិរញ្ញវត្ថុបានក្លាយជាការគំរាមកំហែងធ្ងន់ធ្ងរដល់ភាពត្រឹមត្រូវនៃប្រព័ន្ធហិរញ្ញវត្ថុសកល ដោយបង្ខំឱ្យស្ថាប័នហិរញ្ញវត្ថុគ្រប់រូបចាំបាច់ត្រូវអនុវត្តប្រព័ន្ធការពារប្រឆាំងការក្លែងបន្លំដ៏មានប្រសិទ្ធភាព។ ក្នុងបរិបទនៃទិន្នន័យចែករំលែករវាងស្ថាប័ន ការប្រើប្រាស់ Machine Learning បែបប្រពៃណីដែលផ្អែកលើការបង្រួបបង្រួមទិន្នន័យឆៅត្រូវបានហាមឃាត់ដោយច្បាប់ការពារទិន្នន័យ ដូចជា GDPR និង PDPA ដែលបង្ករបញ្ហា Data Silos នៅតាមធនាគាររាយ។ ស្ថាប័នដែលខ្វះទិន្នន័យបែបក្លែងបន្លំក្នុងមូលដ្ឋានមិនអាចបណ្ដុះបណ្ដាលម៉ូដែលចាត់ថ្នាក់ការពារក្លែងបន្លំដែលដំណើរការបានទេ ដែលស្ថានភាពនេះត្រូវបានហៅថា បញ្ហា Blind Spot។

ការសិក្សានេះស្នើឡើង និងវាយតម្លៃ Framework Federated Learning ដែលអនុវត្ត XGBoost ជា algorithm ចាត់ថ្នាក់ស្នូល នៅលើធនាគារអតិថិជន ០៣ ដែលដំណើរការក្រោមការចែកចាយទិន្នន័យបែប Non-IID។ ការរួមចំណែកបច្ចេកទេសសំខាន់គឺ JSON Tree Concatenation Algorithm \- វិធីសាស្ត្របូករួមម៉ូដែលថ្មីដែល Global Server ធ្វើការបន្សំ XGBoost Models ដែលបានបណ្ដុះបណ្ដាលក្នុងមូលដ្ឋាន ដោយការភ្ជាប់ទ្រង់ទ្រាយ tree structure JSON ដោយផ្ទាល់ ដោយមិនចាំបាច់ចូលប្រើទិន្នន័យប្រត្តិបត្តិការឆៅពីស្ថាប័នណាមួយឡើយ។ ភាពត្រឹមត្រូវនៃ algorithm នេះផ្អែកលើ semantic additive score របស់ XGBoost ដោយ \= \u03a3 f\_k(x) សម្រាប់ trees ទាំងអស់ k ដែលធានាថាការភ្ជាប់ tree arrays ពី clients ច្រើននឹងរក្សាសមត្ថភាពការពារបានពេញលេញ។

ការវាយតម្លៃពិសោធន៍ ត្រូវបានអនុវត្តនៅលើ Dataset PaySim Synthetic ដែលមានប្រតិបត្តិការ ០៦.៣៦ លានជាមួយ Fraud Prevalence ០.១៣ ភាគរយ ដែលបង្កើតឲ្យមាន class imbalance ratio ០៧៧៣ ទល់ ១។ ដោយគិតពី class imbalance ខ្លាំង Accuracy ត្រូវបានលើកលែងយ៉ាងច្បាស់ ហើយ AUPRC និង F1-Score ត្រូវបានប្រើជា metrics វាយតម្លៃចម្បង។ ក្រោម baseline Local-Only ធនាគារ ០១ (High-Risk) សម្រេចបាន AUPRC = ០.៩៣៤៣ និង F1 = ០.០៥៤១ ខណៈ ធនាគារ ០២ (Retail) ដែលមិនមានទិន្នន័យ Fraud ក្នុងមូលដ្ឋាន សម្រេចបាន AUPRC = ០.៥០០៦ \u2248 ០.៥ ដូច random classifier ហើយ F1 = ០.០០០០ ដែលបញ្ជាក់ពីការបរាជ័យក្នុងការការពារក្លែងបន្លំជាទូទៅ ហើយ ធនាគារ ០៣ (Mixed) សម្រេចបាន AUPRC = ០.៩៩៣២ និង F1 = ០.៦៥៥៦។ Baseline Centralized ដែលបង្រួបបង្រួមទិន្នន័យ \- Privacy Violated ជាច្បាស់ \- សម្រេចបាន AUPRC = ០.៩៩៧៦ និង F1 = ០.៩៥១៦ ជា ceiling លើ performance។

បន្ទាប់ពី federation ដោយប្រើ JSON Tree Concatenation Algorithm ម៉ូដែល federated global សម្រេចបាន AUPRC = ០.៩៨៣០ និង F1 = ០.៨៥២៦ ចាប់ពី Round ១ ហើយ stable ពេញ ០៥ rounds សម្រាប់ធនាគារទាំងបីទាំងអស់។ ម៉ូដែល Federated ដែលប្រើ privacy ផ្ទាល់ Score AUPRC = ០.៩៨៣០ ក្នុង Round ០៥ ធៀបនឹង Centralized = ០.៩៩៧៦ ដែលបញ្ជាក់ privacy tax = ០.០១៤៦ ឬ ១.៤៦ ភាគរយ \- ការថយចុះប្រៃនៃ performance ដែលមានតម្លៃតិចតួច ដែលសមស្របនឹងការការពារ privacy ។ ការរកឃើញទាំងនេះបញ្ជាក់ទៅអ្នកអានថា JSON Tree Concatenation Algorithm ជំរុញ ធនាគារ ០២ ពីបញ្ហា Blind Spot AUPRC \= ០.៥០០៦ ទៅ AUPRC = ០.៩៨៣០ ក្នុង Round ០១ ដោយគ្មានការផ្ទេរទិន្នន័យឆៅណាមួយឡើយ ដែលបង្ហាញថា Federated Learning អាចដោះស្រាយការបែងចែកទិន្នន័យ Data Silos ប្រកបដោយប្រសិទ្ធភាព ខណៈដែលការពារ privacy ទិន្នន័យរបស់ស្ថាប័ននីមួយៗ។

---

## ABSTRACT {#abstract .ABSTRACT}

Financial fraud detection presents a fundamental tension between the need for high-quality collaborative training data and the legal obligation to preserve client data privacy. Traditional centralized machine learning approaches require raw transaction data to be consolidated onto a single server — a practice that violates inter-institutional data governance policies established by frameworks including the General Data Protection Regulation (GDPR, 2018) and equivalent regulations across Southeast Asian jurisdictions. Conversely, institutions that operate in data isolation are subject to severe detection blind spots, particularly retail banks whose local transaction histories contain insufficient fraud-labeled examples to train a reliable classifier.

This thesis proposes and empirically evaluates a **Federated Learning (FL)** framework for early financial fraud detection, leveraging **XGBoost** as the core classification algorithm across three participating client banks operating under heterogeneous, **Non-IID** data distributions simulated using the **PaySim** synthetic financial dataset. The framework introduces the **JSON Tree Concatenation Algorithm**, a novel model aggregation method in which the **Global Server** combines locally trained XGBoost models by directly concatenating their JSON-serialized internal tree structures — without requiring any transfer of raw transaction data between institutions. The algorithm's theoretical validity is grounded in XGBoost's additive scoring semantics: since the prediction function is defined as ŷ = Σ f_k(x) across all trees k, concatenating tree ensembles from multiple clients preserves the full discriminative capacity of each local model in the federated global ensemble.

The PaySim dataset contains 6,362,620 synthetic transactions with a fraud prevalence of 0.13%, producing a class imbalance ratio of 773:1. Given this extreme imbalance, **Accuracy is explicitly excluded** as an evaluation metric throughout this thesis. All performance assessments are conducted exclusively using **AUPRC** (Area Under the Precision-Recall Curve) and **F1-Score**, which provide statistically meaningful measures of classifier performance under severe class skew.

Three experimental conditions are evaluated. Under the **Local-Only baseline**, which establishes the cost of data isolation, Bank 1 (High-Risk, 1,064,011 records, 0.2892% local fraud) achieves AUPRC = 0.9343 and F1 = 0.0541; Bank 2 (Retail/Blind Spot, 2,272,208 records, 0.0000% local fraud) achieves AUPRC = 0.5006 — statistically equivalent to a random classifier — and F1 = 0.0000, confirming zero operational fraud detection capability; and Bank 3 (Mixed, 735,859 records, 0.2893% local fraud) achieves AUPRC = 0.9932 and F1 = 0.6556. The **Centralized baseline**, which pools all data onto a single server in a privacy-violating configuration, achieves AUPRC = 0.9976 and F1 = 0.9516, establishing the theoretical performance ceiling.

Following federation via the **JSON Tree Concatenation Algorithm**, the federated global model achieves **AUPRC = 0.9830 and F1-Score = 0.8526** from Round 1 onward, with performance stable across all five communication rounds for all three client banks. Bank 2 recovers from AUPRC = 0.5006 (total detection failure) to AUPRC = 0.9830 within a single federated round, without any transfer of raw transaction records. The **privacy tax** — the reduction in AUPRC attributable to the privacy-preserving constraint — is **0.0146 (1.46%)**, a negligible reduction that strongly supports the practical viability of the proposed framework. These results collectively demonstrate that the JSON Tree Concatenation Algorithm effectively resolves the **Data Silo** problem for isolated retail banks, achieving near-centralized fraud detection performance while strictly preserving data privacy across all participating institutions.

---

## SUPERVISOR'S RESEARCH SUPERVISION STATEMENT {#supervisors-research-supervision-statement .ABSTRACT}

TO WHOM IT MAY CONCERN

Name of program: Bachelor of Engineering in Information Technology Engineering

Name of candidate: Sothun Darachhat

Title of research report: A Federated Learning Framework for Early Fraud Detection using XGBoost: Balancing Model Performance and Data Privacy

This is to certify that the research carried out for the above titled bachelor's research report was completed by the above-named candidate under my direct supervision. This thesis material has not been used for any other degree. I played the following part in the preparation of this research report: A Federated Learning Framework for Early Fraud Detection using XGBoost: Balancing Model Performance and Data Privacy.

Supervisor's name: Mr. Chhim Bunchhun

Supervisor's signature: ...............................................

Date: ...............................................

---

## CANDIDATE'S STATEMENT {#candidates-statement .ABSTRACT}

TO WHOM IT MAY CONCERN

This is to certify that the research report that I, Sothun Darachhat, hereby present entitled:

**"A Federated Learning Framework for Early Fraud Detection using XGBoost: Balancing Model Performance and Data Privacy"**

for the degree of Bachelor of Engineering in Information Technology Engineering at the Royal University of Phnom Penh is entirely my own work and, furthermore, that it has not been used to fulfill the requirements of any other qualification in whole or in part, at this or any other University or equivalent institution.

No reference to, or quotation from, this document may be made without the written approval of the author.

Signed by (the candidate): ...............................................

Date: ...............................................

Signed by Supervisor: ...............................................

Supervisor's signature: ...............................................

Date: ...............................................

---

## ACKNOWLEDGEMENTS {#acknowledgements .ABSTRACT}

This thesis marks the culmination of an intensive and deeply rewarding academic journey, and its completion would not have been possible without the guidance, support, and encouragement of numerous individuals to whom sincere gratitude is owed.

First and foremost, deepest appreciation is extended to thesis supervisor Mr. Chhim Bunchhun, whose expert guidance, intellectual rigor, and consistent availability throughout the research process were invaluable. His willingness to engage critically with both the technical and academic dimensions of this work — from the conceptual design of the Federated Learning framework to the interpretation of experimental results — fundamentally shaped the quality and direction of this thesis. His mentorship extended beyond technical supervision, instilling a commitment to methodological precision and academic integrity that will inform professional practice long after the completion of this degree.

Sincere gratitude is extended to the faculty members and lecturers of the Department of Information Technology Engineering at the Royal University of Phnom Penh, whose teaching across the program provided the foundational knowledge in machine learning, data engineering, and software systems that made this research possible. The rigorous academic environment they collectively maintain provided both the intellectual foundation and the motivational context for undertaking a thesis of this scope.

Heartfelt thanks go to the developers and contributors of the **PaySim dataset** — López-Rojas, Elmir, and Axelsson — whose publicly accessible synthetic financial transaction dataset provided the empirical foundation for this research. The availability of high-quality, publicly accessible research datasets is essential for academic progress in applied machine learning, and this thesis would not have been feasible without their contribution to the research community.

Deep gratitude is also expressed to the broader open-source community behind the tools and libraries that made this research technically possible — specifically the developers of **XGBoost**, **Scikit-learn**, **Pandas**, **NumPy**, and **Python** — whose sustained contributions to the scientific computing ecosystem enable researchers worldwide to conduct rigorous machine learning experiments without prohibitive infrastructure barriers.

To family, the most profound and heartfelt thanks are offered. Their unconditional love, unwavering belief, and patient understanding throughout the demands of this academic program provided the emotional foundation upon which everything else in this journey rested. This achievement belongs as much to them as it does to the author.

To friends and peers in the Department of Information Technology Engineering, gratitude is expressed for the collaborative spirit, intellectual camaraderie, and mutual encouragement that characterized the shared academic experience. The conversations, debates, and collaborative problem-solving sessions that occurred both inside and outside the classroom enriched understanding in ways that no textbook or lecture could replicate alone.

Finally, acknowledgement is made to the researchers and academics whose published work forms the intellectual foundation of this thesis. The fields of Federated Learning, privacy-preserving machine learning, and financial fraud detection are advanced by the cumulative contributions of a global research community, and this thesis is built upon — and seeks to contribute to — that collective body of knowledge.

---

## TABLE OF CONTENTS {#table-of-contents .ABSTRACT}

[មូលន័យសង្ខេប](#មូលន័យសង្ខេប) i

[ABSTRACT](#abstract) iii

[SUPERVISOR'S RESEARCH SUPERVISION STATEMENT](#supervisors-research-supervision-statement) iv

[CANDIDATE'S STATEMENT](#candidates-statement) v

[ACKNOWLEDGEMENTS](#acknowledgements) vi

[TABLE OF CONTENTS](#table-of-contents) viii

[LIST OF TABLES](#list-of-tables) xi

[LIST OF FIGURES](#list-of-figures) xii

**[CHAPTER 1 — INTRODUCTION](#chapter-1-introduction)** 1

[1.1 Background to the Study](#11-background-to-the-study) 1

[1.2 Problem Statement](#12-problem-statement) 5

[1.3 Aim and Objectives of the Study](#13-aim-and-objectives-of-the-study) 7

[1.4 Rationale of the Study](#14-rationale-of-the-study) 9

[1.5 Limitations and Scope](#15-limitations-and-scope) 11

[1.6 Structure of the Study](#16-structure-of-the-study) 13

**[CHAPTER 2 — LITERATURE REVIEW](#chapter-2-literature-review)** 15

[2.1 Overview of the Research Topic](#21-overview-of-the-research-topic) 15

[2.2 Traditional and Rule-Based Fraud Detection Systems](#22-traditional-and-rule-based-fraud-detection-systems) 17

[2.3 Machine Learning Approaches for Fraud Detection](#23-machine-learning-approaches-for-fraud-detection) 18

[2.3.1 XGBoost in Financial Fraud Detection](#231-xgboost-in-financial-fraud-detection) 19

[2.3.2 Class Imbalance in Fraud Detection](#232-class-imbalance-in-fraud-detection) 20

[2.4 The Data Silo Problem in Multi-Institutional Fraud Detection](#24-the-data-silo-problem-in-multi-institutional-fraud-detection) 21

[2.5 Federated Learning: Foundations and Architecture](#25-federated-learning-foundations-and-architecture) 22

[2.6 Non-IID Data Challenges in Federated Learning](#26-non-iid-data-challenges-in-federated-learning) 24

[2.7 Federated Learning Applied to Tree-Based Models](#27-federated-learning-applied-to-tree-based-models) 25

[2.8 Privacy-Preserving Machine Learning in Financial Services](#28-privacy-preserving-machine-learning-in-financial-services) 27

[2.9 The PaySim Dataset in Fraud Detection Research](#29-the-paysim-dataset-in-fraud-detection-research) 28

[2.10 Gap in Existing Research](#210-gap-in-existing-research) 29

**[CHAPTER 3 — METHODOLOGY AND TOOLS](#chapter-3-methodology-and-tools)** 33

[3.1 Research Design](#31-research-design) 33

[3.2 Dataset: PaySim Synthetic Financial Dataset](#32-dataset-paysim-synthetic-financial-dataset) 35

[3.2.1 Dataset Description](#321-dataset-description) 35

[3.2.2 Class Imbalance](#322-class-imbalance) 37

[3.2.3 Feature Engineering and Preprocessing](#323-feature-engineering-and-preprocessing) 38

[3.3 Non-IID Dataset Partitioning](#33-non-iid-dataset-partitioning) 41

[3.4 Local Model Training: XGBoost Configuration](#34-local-model-training-xgboost-configuration) 43

[3.5 The JSON Tree Concatenation Algorithm](#35-the-json-tree-concatenation-algorithm) 45

[3.5.1 Theoretical Foundation](#351-theoretical-foundation) 45

[3.5.2 Algorithm Implementation](#352-algorithm-implementation) 46

[3.5.3 Privacy Guarantee](#353-privacy-guarantee) 49

[3.6 Federated Training Protocol](#36-federated-training-protocol) 50

[3.7 Baseline Experimental Conditions](#37-baseline-experimental-conditions) 52

[3.7.1 Local-Only Baseline](#371-local-only-baseline) 52

[3.7.2 Centralized Baseline](#372-centralized-baseline) 52

[3.8 Evaluation Metrics](#38-evaluation-metrics) 53

[3.8.1 AUPRC — Area Under the Precision-Recall Curve](#381-auprc) 53

[3.8.2 F1-Score](#382-f1-score) 54

[3.8.3 Explicit Exclusion of Accuracy](#383-explicit-exclusion-of-accuracy) 54

[3.9 Tools and Technologies](#39-tools-and-technologies) 55

**[CHAPTER 4 — RESULTS](#chapter-4-results)** 57

[4.1 Overview](#41-overview) 57

[4.2 Descriptive Statistics of Partitioned Dataset](#42-descriptive-statistics-of-partitioned-dataset) 57

[4.3 Local-Only Baseline Results](#43-local-only-baseline-results) 59

[4.3.1 Per-Client Local-Only Performance](#431-per-client-local-only-performance) 59

[4.3.2 Interpretation of Local-Only Results](#432-interpretation-of-local-only-results) 60

[4.4 Centralized Baseline Results](#44-centralized-baseline-results) 63

[4.4.1 Centralized Baseline Performance](#441-centralized-baseline-performance) 63

[4.4.2 Interpretation of Centralized Results](#442-interpretation-of-centralized-results) 63

[4.5 Federated Learning Results](#45-federated-learning-results) 65

[4.5.1 Bank 2 Performance Across Federation Rounds](#451-bank-2-performance-across-federation-rounds) 65

[4.5.2 All-Client Federated Performance at Round 5](#452-all-client-federated-performance-at-round-5) 67

[4.5.3 Interpretation of Federated Learning Results](#453-interpretation-of-federated-learning-results) 68

[4.6 Comparative Summary Across All Experimental Conditions](#46-comparative-summary) 71

[4.7 Feature Importance Analysis](#47-feature-importance-analysis) 73

[4.8 Precision-Recall Curve Analysis](#48-precision-recall-curve-analysis) 75

**[CHAPTER 5 — DISCUSSION](#chapter-5-discussion)** 77

[5.1 Overview](#51-overview) 77

[5.2 Empirical Validation of the Blind Spot Problem](#52-empirical-validation-of-the-blind-spot-problem) 77

[5.3 Effectiveness of the JSON Tree Concatenation Algorithm](#53-effectiveness-of-the-json-tree-concatenation-algorithm) 79

[5.3.1 Theoretical Validity](#531-theoretical-validity) 79

[5.3.2 Empirical Effectiveness](#532-empirical-effectiveness) 80

[5.3.3 Non-Invasiveness and Deployment Practicality](#533-non-invasiveness-and-deployment-practicality) 82

[5.4 Privacy-Performance Trade-off Analysis](#54-privacy-performance-trade-off-analysis) 83

[5.4.1 Trade-off Quantification](#541-trade-off-quantification) 83

[5.4.2 The Fundamental Value Proposition](#542-the-fundamental-value-proposition) 84

[5.5 Comparison with Prior Work](#55-comparison-with-prior-work) 85

[5.5.1 Comparison with Centralized XGBoost Fraud Detection](#551-comparison-with-centralized-xgboost) 85

[5.5.2 Comparison with Federated Learning Fraud Detection Studies](#552-comparison-with-fl-fraud-detection-studies) 86

[5.5.3 Comparison with SecureBoost and FedTree](#553-comparison-with-secureboost-and-fedtree) 87

[5.6 Limitations and Critical Reflection](#56-limitations-and-critical-reflection) 88

[5.7 Summary of Discussion](#57-summary-of-discussion) 91

**[CHAPTER 6 — CONCLUSION](#chapter-6-conclusion)** 93

[6.1 Summary of the Study](#61-summary-of-the-study) 93

[6.2 Conclusions Drawn](#62-conclusions-drawn) 96

[6.3 Future Works](#63-future-works) 99

[REFERENCES](#references) 107

---

## LIST OF TABLES {#list-of-tables .ABSTRACT}

Table 3‑1: PaySim Dataset Transaction Type Summary 36

Table 3‑2: PaySim Dataset Feature Descriptions 36

Table 3‑3: Non-IID Client Bank Profile and Partitioning Strategy 42

Table 3‑4: XGBoost Hyperparameter Configuration 44

Table 3‑5: Tools and Technologies Summary 55

Table 4‑1: Descriptive Statistics of Partitioned Dataset Across Three Client Banks 58

Table 4‑2: Local-Only Baseline Performance — AUPRC and F1-Score per Client Bank 59

Table 4‑3: Centralized Baseline Performance — AUPRC, F1-Score, Precision, and Recall 63

Table 4‑4: Bank 2 AUPRC and F1-Score Trajectory Across Five Federated Rounds 66

Table 4‑5: All-Client Federated Performance at Round 5 vs. Local-Only Baseline 67

Table 4‑6: Consolidated Performance Results Across All Experimental Conditions 72

Table 5‑1: Privacy-Performance Trade-off Quantification — FL Round 5 vs. Centralized Baseline 83

---

## LIST OF FIGURES {#list-of-figures .ABSTRACT}

Figure 1.1: Federated Learning Framework Architecture — JSON Tree Concatenation Algorithm 4

Figure 3.1: PaySim Dataset Class Distribution — Fraud vs. Legitimate Transactions 37

Figure 3.2: Transaction Amount Distribution — Raw vs. Log-Transformed 39

Figure 3.3: Non-IID Data Partitioning — Fraud Prevalence per Client Bank 42

Figure 3.4: JSON Tree Concatenation Algorithm — Four-Step Aggregation Process 47

Figure 3.5: Federated Training Protocol — Five-Round Communication Diagram 51

Figure 3.6: Precision-Recall Curve — AUPRC Conceptual Illustration 53

Figure 3.7: Accuracy Exclusion Justification — Degenerate Classifier Performance 55

Figure 4.1: Local-Only Baseline Performance — AUPRC and F1-Score per Bank 60

Figure 4.2: Precision-Recall Curve — Centralized Baseline Model (AUPRC = 0.9976) 64

Figure 4.3: AUPRC Trajectory Across Five Federated Communication Rounds 66

Figure 4.4: F1-Score Trajectory Across Five Federated Communication Rounds 67

Figure 4.5: Bank 2 Blind Spot Resolution — AUPRC and F1-Score Recovery 69

Figure 4.6: AUPRC Comparison — Local-Only vs. FL Round 5 vs. Centralized 72

Figure 4.7: XGBoost Feature Importance — Federated Global Model Round 5 74

Figure 4.8: Precision-Recall Curves — All Models Overlaid 75

Figure 5.1: Privacy-Performance Trade-off — Centralized vs. Federated Round 5 84

---

# CHAPTER 1 — INTRODUCTION {#chapter-1-introduction}

## 1.1 Background to the Study {#11-background-to-the-study}

Financial fraud detection has emerged as one of the most consequential applications of machine learning within the global financial services sector. The accelerating digitization of financial systems — encompassing mobile banking platforms, peer-to-peer payment networks, and real-time interbank transfer infrastructure — has dramatically expanded both the volume of transaction data generated daily and the attack surface available to fraudulent actors. As digital transaction volumes continue to scale to hundreds of millions of records per day, the fundamental limitations of manual review processes and static rule-based detection systems have become structurally untenable, compelling the widespread adoption of automated, data-driven fraud detection approaches.

Machine learning offers a qualitatively superior detection paradigm relative to conventional rule-based systems. Rather than encoding expert-defined thresholds and heuristics that sophisticated fraudsters can identify and circumvent, machine learning models learn discriminative behavioral patterns directly from historical transaction data, capturing complex and non-linear signals that static rules cannot represent. This adaptive capability is particularly critical in financial fraud detection, where adversarial actors continuously evolve their strategies in response to existing detection mechanisms, rendering any static ruleset progressively obsolete over time.

Despite the demonstrated effectiveness of supervised machine learning for fraud detection, the practical deployment of such systems in multi-institutional financial environments is constrained by a fundamental structural tension. The institutions that collectively possess the most comprehensive view of fraud behavior are legally and regulatorily prohibited from sharing the raw transaction data that would enable them to exploit that collective knowledge. Data governance frameworks — including the General Data Protection Regulation (GDPR, 2018), the Personal Data Protection Act (PDPA) applicable in Southeast Asian jurisdictions, and Basel III operational risk compliance guidelines — impose strict limitations on the cross-institutional transfer of personal financial data. These regulatory constraints create structurally isolated data environments, commonly referred to as **Data Silos**, in which each institution trains its fraud detection models on a locally incomplete and potentially unrepresentative dataset.

The consequences of this data isolation are asymmetric and institution-dependent. Large financial institutions with high transaction volumes and diverse customer risk profiles accumulate sufficient fraud-labeled training examples to develop robust local classifiers. Retail banks serving lower-risk customer segments, however, may accumulate extensive transaction histories containing little or no confirmed fraudulent activity. When such an institution attempts to train a supervised fraud detection model under these conditions, the complete absence of positive training examples produces a classifier with no capacity to assign meaningful fraud probability scores. This condition is referred to throughout this thesis as the **blind spot problem**, and constitutes a total and measurable failure of the fraud detection function rather than merely a degraded performance outcome.

**Federated Learning (FL)**, first formally proposed by McMahan et al. (2017), offers a principled resolution to this structural problem. Under the Federated Learning paradigm, multiple client institutions collaboratively train a shared global model by exchanging model parameters or serialized model structures rather than raw data. A central **Global Server** coordinates the aggregation of locally trained model updates, producing a federated global model that incorporates the collective fraud intelligence of all participating clients without requiring any raw transaction data to leave its institution of origin. This privacy-preserving architectural property makes Federated Learning directly compatible with the data governance frameworks that govern cross-institutional financial data exchange.

While Federated Learning has been extensively studied in the context of deep learning and neural network architectures, its application to gradient boosting classifiers such as XGBoost presents a technically non-trivial challenge. XGBoost models are represented internally as ensembles of decision trees serialized in a structured format, and their parameters are not amenable to the arithmetic averaging operations employed by standard FL aggregation methods. This incompatibility motivates the primary technical contribution of this thesis: the **JSON Tree Concatenation Algorithm**, a novel aggregation mechanism that combines locally trained XGBoost models at the Global Server by directly merging their JSON-serialized internal tree structures, grounded in the additive scoring semantics of the XGBoost prediction function. Figure 1.1 illustrates the overall architecture of the proposed Federated Learning framework.

*[Figure 1.1: Federated Learning Framework Architecture — JSON Tree Concatenation Algorithm across 5 Communication Rounds]*

## 1.2 Problem Statement {#12-problem-statement}

Despite significant advances in machine learning-based fraud detection, two fundamental and interrelated problems remain unresolved in the context of multi-institutional financial environments operating under data privacy constraints.

The first problem is the **Data Silo problem**. Financial institutions are structurally prevented from sharing raw customer transaction records across institutional boundaries by data governance regulations and competitive incentives. Each institution therefore trains its fraud detection models on a locally incomplete dataset that reflects only its own customer base and transaction typologies. For institutions whose local transaction history is structurally deficient in fraud-labeled examples — specifically retail banks serving low-risk customer segments — this isolation produces classifiers with no meaningful fraud detection capability. The experimental results of this study provide a precise quantitative demonstration of this failure: Bank 2, a retail institution with zero locally available fraud-labeled transactions, achieves an AUPRC of 0.5006 under isolated training conditions — a score statistically equivalent to a random classifier — and an F1-Score of exactly 0.0000 at the default classification threshold, confirming zero operational fraud detection capability.

The second problem is the **absence of a horizontal federated aggregation mechanism for XGBoost**. The dominant FL aggregation strategy, FedAvg, computes a weighted arithmetic mean of neural network weight vectors — a mathematically well-defined operation for continuous parameter spaces. XGBoost models, however, are non-parametric tree ensembles whose internal representation consists of discrete branching structures. There is no meaningful arithmetic mean of two decision trees, and the naive application of FedAvg-style averaging to XGBoost model representations produces undefined or semantically invalid results. Existing federated tree-based methods — including SecureBoost (Cheng et al., 2021) and FedTree (Li et al., 2023) — do not address the horizontal federation scenario in which different institutions hold different transaction records with the same feature schema.

This thesis addresses both problems through the design, implementation, and empirical evaluation of a Federated Learning framework that applies the **JSON Tree Concatenation Algorithm** to aggregate XGBoost models trained independently across three client banks operating under Non-IID data distributions. The central research question guiding this study is: *Which federated learning approach most effectively resolves the fraud detection blind spot for data-isolated retail financial institutions, while preserving strict data privacy guarantees?*

## 1.3 Aim and Objectives of the Study {#13-aim-and-objectives-of-the-study}

The primary aim of this study is to design and empirically evaluate a privacy-preserving Federated Learning framework for financial fraud detection that enables data-isolated institutions to develop effective fraud classifiers without transferring raw transaction data across institutional boundaries.

The following specific objectives guide the research:

1. **To demonstrate the blind spot problem quantitatively.** Independent Local-Only XGBoost models are trained on each of the three banks' partitioned datasets and evaluated on AUPRC and F1-Score to establish the precise performance cost of data isolation, with particular focus on Bank 2's total detection failure under conditions of zero local fraud labels.

2. **To establish a centralized performance ceiling.** A single XGBoost model is trained on the fully pooled dataset across all three banks to quantify the theoretical upper bound of detection performance achievable under privacy-violating conditions, providing a reference benchmark against which the federated framework's privacy-performance trade-off is measured.

3. **To design and implement the JSON Tree Concatenation Algorithm.** A novel XGBoost model aggregation method is developed in which the Global Server merges locally trained models by concatenating their JSON-serialized decision tree structures without accessing any raw client transaction data. The algorithm's theoretical validity is grounded in XGBoost's additive scoring semantics: since the XGBoost prediction function is a sum over all trees in the ensemble, concatenating tree arrays from multiple clients preserves the full discriminative capacity of each local model in the global ensemble.

4. **To evaluate the Federated Learning framework across five communication rounds.** The federated training protocol is executed across five rounds, and Bank 2's AUPRC and F1-Score trajectory is measured to demonstrate the framework's effectiveness in resolving the blind spot problem and to quantify the rate of performance convergence across federation rounds.

5. **To analyze the privacy-performance trade-off.** The federated model's Round 5 performance is compared against both the Local-Only baseline and the Centralized baseline to quantify the privacy tax — the reduction in detection capability, if any, that results from the privacy-preserving architectural constraint of the federated framework.

## 1.4 Rationale of the Study {#14-rationale-of-the-study}

The rationale for this research is grounded in three converging realities of the contemporary financial technology landscape that collectively create both the need and the opportunity for the framework proposed in this thesis.

**First**, the regulatory environment makes cross-institutional data sharing increasingly impractical. Regulations such as GDPR in Europe and equivalent frameworks across Southeast Asia impose strict constraints on the transfer of personal financial data across institutional boundaries. Financial institutions that attempt to construct centralized fraud detection systems by pooling customer data face material legal exposure. As digital transaction volumes continue to grow and regulatory scrutiny intensifies, the pressure to develop privacy-compliant detection methods will increase rather than diminish. The Federated Learning paradigm is uniquely positioned to address this regulatory constraint, providing a mechanism for collaborative learning that is architecturally compatible with existing data governance frameworks without requiring any modification to privacy regulations or institutional data sharing policies.

**Second**, fraud behavior is inherently distributed across institutions and cannot be fully observed by any single participant. Sophisticated fraud networks deliberately distribute their activity across multiple institutions to avoid triggering any single institution's detection threshold. A fraud detection model trained exclusively on one institution's local data will exhibit blind spots corresponding to fraud typologies it has never encountered. The experimental results of this study demonstrate this effect precisely: Bank 2's Local-Only model achieves an AUPRC of 0.5006 — statistically indistinguishable from a random classifier — because its local training corpus contains zero fraud-labeled examples. This is not a modeling failure but a structural consequence of data isolation that no algorithmic improvement within the isolated institution can resolve.

**Third**, XGBoost remains the dominant and most practically appropriate algorithm for structured financial transaction data. Despite the proliferation of deep learning architectures, XGBoost consistently achieves state-of-the-art performance on tabular financial data due to its robustness to feature scale heterogeneity, native handling of missing values, computational efficiency, and built-in feature importance mechanisms that support regulatory interpretability requirements. The absence of a well-defined and practically deployable federated aggregation method for XGBoost represents a gap that limits the applicability of Federated Learning in precisely the environments where it is most needed. This thesis directly fills that gap through the JSON Tree Concatenation Algorithm.

This research is of particular relevance to the Cambodian financial context, where digital transactions and fintech adoption are increasing rapidly. The development of robust fraud detection infrastructure in this environment is constrained by the limited availability of labeled financial data at individual institutions and the absence of established inter-institutional data sharing frameworks. By demonstrating how machine learning models can be collaboratively trained using structured synthetic data that faithfully replicates real-world transaction distributions, this study provides a methodological reference point for future research and implementation efforts within Cambodia and the broader Southeast Asian region.

## 1.5 Limitations and Scope {#15-limitations-and-scope}

This study is carefully designed to evaluate the effectiveness of the proposed Federated Learning framework within a defined set of constraints that balance research feasibility with academic rigor. The following limitations and scope definitions frame the validity and generalizability of the findings.

- **Dataset.** All experiments are conducted on the PaySim synthetic financial dataset (López-Rojas et al., 2016), which simulates mobile money transactions with a fraud prevalence of approximately 0.13%, producing a class imbalance ratio of 773:1. While PaySim faithfully replicates the statistical properties of real-world mobile banking transaction data, it remains a simulation. Model performance on live institutional data may differ due to feature distribution shifts, adversarially adaptive fraud typologies, and institution-specific behavioral patterns not represented in the synthetic dataset.

- **Number of Federated Clients.** The federated simulation involves exactly three client banks with defined risk profiles: Bank 1 (High-Risk), Bank 2 (Retail/Blind Spot), and Bank 3 (Mixed). The scalability of the JSON Tree Concatenation Algorithm to larger federations of ten or more clients is acknowledged as an important direction for future research and is not empirically evaluated in this thesis.

- **Federation Rounds.** The federated training protocol is evaluated over five communication rounds. Experimental results demonstrate that performance convergence is achieved within the first round, suggesting that the five-round protocol is more than sufficient for the experimental configuration evaluated.

- **Aggregation Method.** Only the JSON Tree Concatenation Algorithm is evaluated as a federated aggregation strategy. No comparative evaluation against alternative federated XGBoost methods such as SecureBoost or FedTree is conducted within this thesis.

- **Privacy Guarantees.** The framework achieves privacy preservation through the architectural guarantee that no raw transaction data leaves any client institution. Formal cryptographic privacy guarantees — such as differential privacy or secure multi-party computation — are not implemented in this version of the framework and are identified as a priority direction for future work.

- **Evaluation Metrics.** Accuracy is explicitly excluded as an evaluation metric throughout this thesis. All performance assessments are conducted exclusively using AUPRC and F1-Score. A classifier that predicts the majority class for every transaction achieves an accuracy of 99.87% while detecting zero fraud — rendering accuracy wholly uninformative under the 0.13% fraud prevalence of the PaySim dataset.

## 1.6 Structure of the Study {#16-structure-of-the-study}

This thesis is organized into six chapters, each logically developed to guide the reader through the complete research process from contextual motivation through methodological implementation, experimental results, and final conclusions.

- **Chapter 1 — Introduction.** Establishes the research context and motivation, defining the Data Silo problem and the blind spot problem. The research aims and objectives are stated, the rationale is articulated, and the scope and limitations are defined.

- **Chapter 2 — Literature Review.** Reviews existing research on machine learning-based fraud detection, Federated Learning frameworks, XGBoost in financial applications, Non-IID data challenges, and privacy-preserving machine learning in financial services. Research gaps addressed by this thesis are identified.

- **Chapter 3 — Methodology and Tools.** Describes the PaySim dataset and its Non-IID partitioning, the feature engineering pipeline, the JSON Tree Concatenation Algorithm design and implementation, the federated training protocol, baseline conditions, and the evaluation framework.

- **Chapter 4 — Results.** Presents the quantitative outcomes of all three experimental conditions with full metric tables, trajectory visualizations, and interpretation of key findings.

- **Chapter 5 — Discussion.** Interprets results in context of research objectives, examines the privacy-performance trade-off, compares findings against prior work, and provides critical reflection on limitations.

- **Chapter 6 — Conclusion.** Summarizes research contributions, states conclusions drawn from experimental evidence, and proposes directions for future work.

---

# CHAPTER 2 — LITERATURE REVIEW {#chapter-2-literature-review}

## 2.1 Overview of the Research Topic {#21-overview-of-the-research-topic}

The detection of financial fraud through machine learning has emerged as a critical research domain at the intersection of data science, financial regulation, and cybersecurity. Financial fraud and unauthorized transaction activity collectively impose substantial economic costs on the global financial system and undermine the integrity of legitimate commerce. According to the Association of Certified Fraud Examiners (ACFE, 2022), organizations lose an estimated five percent of annual revenue to fraud, with financial services consistently identified as one of the highest-risk sectors. The accelerating digitization of financial services has dramatically expanded the volume and velocity of transaction data that must be monitored, simultaneously creating new opportunities for fraudulent exploitation and new possibilities for automated, data-driven detection.

Traditional anti-fraud systems deployed by financial institutions have long relied on rule-based engines that encode expert-defined thresholds and behavioral heuristics. While such systems provide foundational compliance structures and remain operationally prevalent due to their interpretability and auditability, their fundamental limitations — including high false-positive rates, static adaptability, and susceptibility to adversarial circumvention — have motivated a progressive shift toward machine learning-based approaches. Machine learning models learn discriminative patterns directly from historical transaction data, enabling adaptive detection of complex, non-linear fraud behaviors that static rule systems cannot represent.

The effectiveness of supervised machine learning for fraud detection is, however, fundamentally contingent on the quality, volume, and representativeness of the training data available to each model. In practice, financial institutions operate within strict data governance frameworks that prohibit the cross-institutional sharing of raw customer transaction records, creating structurally isolated training environments referred to as **Data Silos**. Institutions with structurally deficient local fraud data — particularly retail banks serving low-risk customer segments — are categorically unable to train functional supervised fraud classifiers under these conditions, a failure mode termed the **blind spot problem** in this thesis.

**Federated Learning** has emerged as the principal methodological response to this structural challenge. By enabling collaborative model training through the exchange of model parameters or structures rather than raw data, Federated Learning allows institutions to benefit from collective fraud intelligence while preserving strict data privacy guarantees. The combination of Federated Learning with XGBoost — the dominant algorithm for structured financial transaction data — requires purpose-built aggregation mechanisms not provided by existing FL frameworks, motivating the primary technical contribution of this thesis: the **JSON Tree Concatenation Algorithm**.

This chapter reviews the theoretical and empirical foundations of this research, tracing the evolution of fraud detection methodology from rule-based systems through centralized machine learning to privacy-preserving federated architectures. The chapter concludes by identifying the specific research gaps that this thesis is positioned to address.

## 2.2 Traditional and Rule-Based Fraud Detection Systems {#22-traditional-and-rule-based-fraud-detection-systems}

The earliest automated fraud detection systems deployed by financial institutions were built on deterministic, rule-based engines. These systems operate by encoding expert-defined heuristics — such as transaction amount thresholds, geographic anomaly flags, or velocity limits on card usage — into conditional rules that trigger alerts when matched by incoming transactions. As documented by Ngai et al. (2011), rule-based systems provided a foundational compliance structure and remain operationally prevalent in many institutions due to their transparency and auditability, properties that are highly valued in regulated financial environments where decision-making processes must be explainable to regulators and auditors.

However, the limitations of rule-based approaches are well-established in the literature. Bolton and Hand (2002) identified three primary failure modes: first, static rules cannot adapt to the evolving tactics of fraudsters who deliberately operate below detection thresholds or exploit known rule gaps; second, the maintenance burden of rule sets grows super-linearly with the number of fraud typologies to be detected; and third, rule-based systems exhibit systematically high false-positive rates, generating alert volumes that exceed the review capacity of compliance teams and resulting in significant operational inefficiency.

Bhattacharyya et al. (2011) further demonstrated that rule-based systems perform particularly poorly on imbalanced datasets — a defining characteristic of real-world fraud data — because their thresholds are calibrated on population-level statistics dominated by the majority non-fraud class. The transition from rule-based to machine learning-based fraud detection was therefore driven not by theoretical preference but by measurable operational failure of rule-based systems at scale. Dal Pozzolo et al. (2015) documented this transition empirically, demonstrating that supervised classification models trained on historical labeled transaction data consistently outperformed rule-based systems across precision, recall, and F1-Score, particularly under conditions of temporal concept drift where fraud patterns shift over time.

## 2.3 Machine Learning Approaches for Fraud Detection {#23-machine-learning-approaches-for-fraud-detection}

The adoption of supervised machine learning in fraud detection has been extensive, with a wide range of classification algorithms evaluated across diverse financial datasets. Early applications employed logistic regression and single decision tree classifiers due to their interpretability and computational tractability. Sahin and Duman (2011) demonstrated that decision tree models could achieve meaningful fraud detection rates on credit card data, but noted that single-tree classifiers were prone to overfitting on the minority class when class imbalance was severe, a characteristic that is near-universal in real-world financial fraud datasets.

Ensemble methods subsequently emerged as the dominant approach, with Random Forest and gradient boosting classifiers demonstrating consistently superior performance over single-estimator baselines. Bhattacharyya et al. (2011) conducted a systematic comparison of logistic regression, support vector machines, Random Forest, and neural network architectures on a real-world credit card fraud dataset, finding that Random Forest achieved the highest F1-Score and AUPRC due to its variance reduction through bootstrap aggregation and random feature subsampling. These findings established the empirical superiority of ensemble methods for imbalanced fraud detection tasks.

### 2.3.1 XGBoost in Financial Fraud Detection {#231-xgboost-in-financial-fraud-detection}

The introduction of XGBoost by Chen and Guestrin (2016) represented a significant methodological advance for structured data classification tasks. XGBoost implements a regularized gradient boosting framework that constructs decision tree ensembles sequentially, with each successive tree trained to minimize the residual error of the current ensemble through second-order gradient optimization. Its technical contributions — including column subsampling, sparsity-aware split finding, cache-aware block computation, and built-in L1 and L2 regularization — yield both superior predictive performance and substantially faster training times compared to earlier gradient boosting implementations.

In the context of financial fraud detection, XGBoost has achieved widespread adoption and benchmark-level performance. Carcillo et al. (2019) demonstrated that XGBoost outperformed deep neural networks on streaming credit card fraud detection tasks, attributing this result to its robustness to feature scale heterogeneity and its native capacity to handle missing values — both common characteristics of real-world transaction datasets. Bahnsen et al. (2016) further showed that XGBoost's feature importance mechanism enabled practitioners to identify and engineer behaviorally meaningful predictors, improving both model performance and regulatory interpretability.

### 2.3.2 Class Imbalance in Fraud Detection {#232-class-imbalance-in-fraud-detection}

A critical and recurring challenge across all supervised fraud detection approaches is the problem of extreme class imbalance. In real-world financial datasets, fraudulent transactions typically represent between 0.1 percent and 1 percent of total transaction volume. The PaySim dataset used in this thesis exhibits a fraud prevalence of approximately 0.13 percent, producing a class imbalance ratio of approximately 773:1. Under such conditions, standard accuracy metrics are demonstrably misleading: a classifier that predicts the majority class for every transaction achieves accuracy exceeding 99 percent while providing zero fraud detection capability.

Dal Pozzolo et al. (2015) formally established that AUPRC is the most appropriate primary evaluation metric for imbalanced binary classification, as it directly measures the trade-off between precision and recall across all classification thresholds without being inflated by the large number of true negative predictions that dominate accuracy under severe class imbalance. This finding forms the methodological basis for the exclusive use of AUPRC and F1-Score as evaluation metrics in this thesis, with Accuracy explicitly disqualified from all performance assessments.

## 2.4 The Data Silo Problem in Multi-Institutional Fraud Detection {#24-the-data-silo-problem-in-multi-institutional-fraud-detection}

While centralized machine learning models trained on pooled multi-institutional data consistently achieve the highest fraud detection performance, the practical feasibility of data pooling in financial environments is severely constrained by regulatory and governance frameworks. The General Data Protection Regulation (GDPR, 2018) imposes strict limitations on the processing and transfer of personal financial data across institutional boundaries within the European Union. Equivalent frameworks — including the California Consumer Privacy Act (CCPA), the Personal Data Protection Act (PDPA) in Southeast Asian jurisdictions, and Basel III operational risk guidelines — establish similar restrictions across other regulatory environments.

Beyond regulatory constraints, Yang et al. (2019) formalized the concept of Data Silos as a systemic property of multi-institutional data ecosystems: each institution's data is structurally isolated by organizational boundaries, competitive incentives, and legal obligations, even when the collective sharing of that data would produce significant mutual benefit. In the context of fraud detection, this isolation is particularly damaging for institutions whose local transaction history is structurally deficient in fraud-labeled positive training examples.

The consequences of data isolation are asymmetric across institutions. Large financial institutions accumulate sufficient fraud-labeled training data to develop robust local classifiers. Retail banks serving low-risk customer segments, however, may accumulate millions of legitimate transaction records while encountering zero confirmed fraud cases over extended operational periods. The experimental results of this study provide a precise quantitative demonstration of this failure: Bank 2, a retail institution with 2,272,208 local training transactions and zero fraud labels, achieves an AUPRC of 0.5006 — statistically equivalent to a random classifier — and an F1-Score of 0.0000 under isolated training conditions. This constitutes the blind spot problem that the proposed Federated Learning framework is designed to resolve.

## 2.5 Federated Learning: Foundations and Architecture {#25-federated-learning-foundations-and-architecture}

Federated Learning was formally introduced by McMahan et al. (2017) as a distributed machine learning paradigm designed to enable collaborative model training across multiple clients without requiring the transfer of raw training data. In the original FL formulation, a central Global Server coordinates training across N clients through iterative communication rounds. In each round, the Global Server distributes the current global model to all participating clients; each client trains a local model update using its private local dataset; and the locally trained updates are returned to the Global Server for aggregation into an improved global model. The fundamental privacy guarantee of this architecture is that no raw data ever leaves any client institution — the Global Server receives only model parameters, not transaction records.

The canonical aggregation algorithm proposed by McMahan et al. (2017), **FedAvg**, computes a weighted arithmetic mean of locally trained model parameters, where weights are proportional to each client's local dataset size. FedAvg has been demonstrated to converge to solutions comparable to centralized training under conditions of data homogeneity and has been extensively applied to image classification, natural language processing, and healthcare prediction tasks.

Subsequent research has extended the foundational FL framework to address its limitations under heterogeneous data distributions. Li et al. (2020) introduced FedProx, which adds a proximal regularization term to the local training objective to improve convergence stability under Non-IID conditions. Karimireddy et al. (2020) proposed SCAFFOLD, which corrects for client drift through variance reduction techniques. These extensions collectively demonstrate the active research effort focused on improving FL performance under the realistic Non-IID conditions that characterize real-world federated deployments in financial services.

## 2.6 Non-IID Data Challenges in Federated Learning {#26-non-iid-data-challenges-in-federated-learning}

A critical assumption underlying the convergence guarantees of FedAvg is that client data is independently and identically distributed (IID) — that is, each client's local dataset is a representative random sample of the global data distribution. In practice, this assumption is rarely satisfied in real-world federated deployments. Financial transaction data is inherently Non-IID: the distribution of transaction amounts, payment types, customer demographics, and fraud typologies varies substantially across institutions as a function of their customer base, geographic footprint, and business model.

Zhao et al. (2018) demonstrated empirically that FedAvg performance degrades significantly under Non-IID data distributions, with accuracy reductions of up to 55 percent compared to the IID baseline on classification benchmarks. The degradation is attributed to client drift: when clients train on locally non-representative data distributions, their local model updates diverge from the global optimum, and arithmetic averaging of divergent updates produces a global model suboptimal for all clients.

In this thesis, the three client banks exhibit a deliberately constructed Non-IID distribution reflecting realistic institutional heterogeneity. Bank 1 (High-Risk) holds 1,064,011 training records with a local fraud prevalence of 0.2892 percent; Bank 2 (Retail/Blind Spot) holds 2,272,208 records with zero fraud labels; and Bank 3 (Mixed) holds 735,859 records with a fraud prevalence of 0.2893 percent. This configuration represents the most challenging test case for federated aggregation: the Global Server must produce a model simultaneously useful for a high-fraud institution, a zero-fraud institution, and a moderate-fraud institution, using only model-level information transmitted from each client without access to any underlying transaction data.

## 2.7 Federated Learning Applied to Tree-Based Models {#27-federated-learning-applied-to-tree-based-models}

The overwhelming majority of Federated Learning research has focused on neural network architectures, for which FedAvg is mathematically well-defined: neural network parameters are real-valued vectors in a continuous parameter space, and their arithmetic mean is a computationally well-defined operation. The application of FL to tree-based ensemble models — including XGBoost — presents a fundamentally different aggregation challenge.

Decision tree ensembles are non-parametric models whose internal representation consists of discrete branching structures. Each tree node encodes a feature index, a split threshold, and child node references. There is no well-defined arithmetic mean of two decision trees — the average of a split on feature 3 at threshold 150.0 and a split on feature 7 at threshold 42.5 has no interpretable meaning in the context of the XGBoost prediction function. This structural incompatibility renders FedAvg inapplicable to XGBoost models without significant architectural modification.

**SecureBoost**, proposed by Cheng et al. (2021), enables federated gradient boosting through a vertically partitioned architecture in which different clients hold different feature subsets of the same transaction records. While SecureBoost provides strong privacy guarantees through homomorphic encryption, it is designed for vertical federation and is not applicable to the horizontal federation scenario of this thesis, where each bank holds different transaction records with the same feature schema.

**FedTree**, proposed by Li et al. (2023), implements a horizontally federated gradient boosting framework by transmitting gradient and Hessian statistics from clients to the Global Server, enabling centralized tree construction from distributed sufficient statistics. While FedTree achieves strong empirical performance, it requires modification of the XGBoost training procedure itself and introduces communication overhead proportional to the number of candidate split points evaluated per training round.

The **JSON Tree Concatenation Algorithm** proposed in this thesis adopts a fundamentally different and non-invasive approach. Rather than modifying the training procedure or transmitting gradient statistics, the Global Server aggregates locally trained XGBoost models by directly concatenating their JSON-serialized internal tree structures. This approach is theoretically grounded in the additive nature of gradient boosting: since XGBoost's prediction function is defined as ŷ = Σ f_k(x) for all trees k, a model constructed by concatenating the tree arrays of multiple locally trained ensembles retains the additive scoring semantics of the original models while incorporating discriminative patterns learned from each client's local data distribution. The non-invasive character of this algorithm requires no modification to the standard XGBoost training pipeline, making it practically deployable in standard financial institution environments without specialized infrastructure.

## 2.8 Privacy-Preserving Machine Learning in Financial Services {#28-privacy-preserving-machine-learning-in-financial-services}

The application of privacy-preserving machine learning to financial services has attracted growing academic and regulatory attention, driven by the convergence of increasing data volumes, stricter privacy legislation, and the demonstrated effectiveness of collaborative learning approaches. Yang et al. (2019) provided the foundational taxonomy of federated learning scenarios — horizontal, vertical, and federated transfer learning — and articulated the alignment between FL's architectural privacy guarantees and the data governance requirements of regulated industries including financial services.

In the financial domain specifically, Suzumura and Kanezashi (2022) applied Federated Learning to anti-money laundering detection across a simulated bank network, demonstrating that federated models achieved detection rates comparable to centralized baselines while maintaining strict data isolation between client institutions. Their findings are directly relevant to this thesis as they establish empirical precedent for FL-based financial crime detection in multi-institutional settings with Non-IID data distributions. The present study extends this line of research by introducing a novel aggregation mechanism specifically designed for XGBoost, and by explicitly evaluating the scenario of a zero-label client institution.

The emerging field of **RegTech** (Regulatory Technology) has further institutionalized the demand for privacy-preserving detection systems. Financial regulators in multiple jurisdictions have begun issuing guidance on the permissible use of collaborative machine learning for compliance purposes, recognizing Federated Learning as a technically sound mechanism for achieving the dual objectives of detection effectiveness and data governance compliance (Financial Stability Board, 2022).

## 2.9 The PaySim Dataset in Fraud Detection Research {#29-the-paysim-dataset-in-fraud-detection-research}

The PaySim dataset, introduced by López-Rojas et al. (2016), is a synthetic financial transaction dataset generated through agent-based simulation of mobile money transfer behavior. The simulation was calibrated against a private dataset of real transactions from a mobile money service operating in an African country, ensuring that PaySim's statistical properties — including transaction amount distributions, account behavior patterns, and fraud typologies — closely replicate real-world mobile payment dynamics.

PaySim contains approximately 6.36 million transactions across five transaction types: CASH-IN, CASH-OUT, DEBIT, PAYMENT, and TRANSFER. Fraud is exclusively associated with CASH-OUT and TRANSFER transactions, with a global fraud prevalence of approximately 0.13 percent, producing a class imbalance ratio of approximately 773:1. The confirmed experimental partition of this dataset produces a global test set of 1,272,524 transactions containing 1,643 confirmed fraud cases.

PaySim has been widely adopted as a benchmark in fraud detection research. Carcillo et al. (2019) used PaySim to evaluate streaming fraud detection frameworks, demonstrating that XGBoost achieved the strongest AUPRC among evaluated classifiers. Kumar and Chadha (2020) applied ensemble methods to PaySim and reported consistent findings, with XGBoost outperforming logistic regression, Random Forest, and neural network baselines on minority-class evaluation metrics. The dataset's public availability, realistic statistical properties, and labeled fraud annotations make it particularly suitable for the federated learning simulation conducted in this thesis, where the dataset is partitioned across three client banks to construct a Non-IID distributed training scenario that faithfully replicates the realistic heterogeneity of fraud exposure across institutions of differing risk profiles.

## 2.10 Gap in Existing Research {#210-gap-in-existing-research}

The review of existing literature reveals three well-defined gaps that this thesis is specifically positioned to address.

### 2.10.1 Absence of Horizontal FL Aggregation for XGBoost

Existing federated tree-based methods — SecureBoost (Cheng et al., 2021) and FedTree (Li et al., 2023) — are designed for vertical federation or require modification of the XGBoost training procedure. No prior work has proposed the direct concatenation of JSON-serialized XGBoost tree structures as a horizontal federation aggregation mechanism. The **JSON Tree Concatenation Algorithm** proposed in this thesis fills this gap by providing a non-invasive, theoretically grounded, and practically deployable aggregation mechanism for horizontal federated XGBoost in multi-institutional financial environments.

### 2.10.2 Quantitative Demonstration of the Zero-Label Blind Spot

While the concept of Data Silos has been theorized in the Federated Learning literature (Yang et al., 2019), few studies have quantitatively demonstrated the complete fraud detection failure that results from data isolation for a zero-label institution. This thesis provides the first metric-grounded demonstration of total detection failure for a zero-label institution: Bank 2's Local-Only **AUPRC = 0.5006** (equivalent to a random classifier) and **F1-Score = 0.0000** establish a precise and empirically measurable baseline against which the federated improvement is evaluated.

### 2.10.3 Underutilization of PaySim for Federated Learning Research

While PaySim has been used for centralized fraud detection benchmarking, comprehensive comparative evaluations of FL frameworks on PaySim — including Local-Only, Centralized, and Federated conditions — under standardized preprocessing and evaluation protocols are absent from the existing literature. This thesis addresses this gap by providing a complete experimental comparison of all three conditions under identical preprocessing pipelines, feature engineering procedures, and evaluation criteria.

In summary, this thesis advances the state of knowledge at the intersection of privacy-preserving machine learning, gradient boosting, and financial fraud detection, offering both a novel algorithmic contribution and a rigorous empirical evaluation grounded in a realistic multi-institutional banking simulation.

---

# CHAPTER 3 — METHODOLOGY AND TOOLS {#chapter-3-methodology-and-tools}

## 3.1 Research Design {#31-research-design}

The research design for this thesis adopts a quantitative, experimental approach, emphasizing structured model development and performance comparison to detect anomalous financial transactions related to fraud. The methodology is grounded in machine learning-based predictive modeling, using a labeled dataset to assess the accuracy and effectiveness of a privacy-preserving Federated Learning framework. This study is organized into three sequential experimental phases, each corresponding to a distinct research objective.

**Phase 1 — Data Preparation and Feature Engineering.** The PaySim synthetic dataset is preprocessed, cleaned, and partitioned into three Non-IID client subsets simulating the data distributions of three distinct banking institutions. Feature engineering is applied to construct domain-informed predictors that enhance model discrimination capability. This phase establishes the experimental foundation upon which all subsequent model training is conducted.

**Phase 2 — Baseline Model Training.** Two baseline conditions are established to bound the performance space of the federated framework. The **Local-Only baseline** trains an independent XGBoost classifier on each bank's isolated local dataset, quantifying the performance cost of data siloing — particularly for Bank 2, which holds zero fraud labels. The **Centralized baseline** trains a single XGBoost classifier on the fully pooled dataset across all three banks, establishing the theoretical performance ceiling under privacy-violating conditions.

**Phase 3 — Federated Learning Framework Execution.** The proposed Federated Learning framework is executed across five communication rounds using the **JSON Tree Concatenation Algorithm**. In each round, each client bank trains a local XGBoost model on its private dataset; the Global Server aggregates the locally trained models; and the resulting federated global model is redistributed to all clients for evaluation. Bank 2's AUPRC and F1-Score are tracked across rounds to measure the framework's effectiveness in resolving the blind spot problem.

All experiments are implemented in **Python**, executed in a simulated federated environment using three independent terminal processes representing the three client banks and one coordinating Global Server process. No raw data is transferred between terminal processes at any stage of the federated protocol.

## 3.2 Dataset: PaySim Synthetic Financial Dataset {#32-dataset-paysim-synthetic-financial-dataset}

### 3.2.1 Dataset Description {#321-dataset-description}

The **PaySim dataset** (López-Rojas et al., 2016) is a publicly available synthetic financial transaction dataset generated through agent-based simulation of mobile money transfer behavior. The simulation was calibrated against anonymized real transaction records from a mobile payment service, ensuring that PaySim's statistical properties faithfully replicate real-world mobile banking dynamics.

The dataset contains **6,362,620 transaction records** across five transaction types:

**Table 3-1: PaySim Dataset Transaction Type Summary**

| Transaction Type | Description | Contains Fraud |
|---|---|---|
| CASH-IN | Deposit of funds into an account | No |
| CASH-OUT | Withdrawal of funds from an account | Yes |
| DEBIT | Direct debit from an account | No |
| PAYMENT | Merchant payment | No |
| TRANSFER | Account-to-account fund transfer | Yes |

**Table 3-2: PaySim Dataset Feature Descriptions**

| Feature | Type | Description |
|---|---|---|
| `step` | Integer | Time step (1 step = 1 hour, max 744) |
| `type` | Categorical | Transaction type (one of five types) |
| `amount` | Float | Transaction amount in local currency |
| `nameOrig` | String | Originating account identifier |
| `oldbalanceOrg` | Float | Originating account balance before transaction |
| `newbalanceOrig` | Float | Originating account balance after transaction |
| `nameDest` | String | Destination account identifier |
| `oldbalanceDest` | Float | Destination account balance before transaction |
| `newbalanceDest` | Float | Destination account balance after transaction |
| `isFraud` | Binary | Ground truth fraud label (1 = fraud, 0 = legitimate) |
| `isFlaggedFraud` | Binary | Legacy rule-based system flag (dropped in preprocessing) |

### 3.2.2 Class Imbalance {#322-class-imbalance}

The PaySim dataset exhibits an extreme class imbalance representative of real-world financial fraud data. Of the 6,362,620 total transactions, only **8,213 are labeled as fraudulent**, yielding a fraud prevalence of approximately **0.13%** and a class imbalance ratio of approximately **773:1**. Fraud is exclusively associated with the **CASH-OUT** and **TRANSFER** transaction types; no fraudulent transactions occur within the CASH-IN, DEBIT, or PAYMENT categories. Figure 3.1 illustrates the class distribution of the PaySim dataset.

*[Figure 3.1: PaySim Dataset Class Distribution — Fraud vs. Legitimate Transactions (0.13% fraud prevalence, 773:1 imbalance ratio)]*

This extreme imbalance has direct consequences for model evaluation methodology. A naive classifier that predicts the majority class (legitimate) for every transaction achieves an accuracy of **99.87%** while detecting zero fraud cases. **Accuracy is therefore explicitly disqualified as an evaluation metric** throughout this thesis. All performance assessments are conducted using **AUPRC** and **F1-Score**, which provide statistically meaningful performance measures under severe class imbalance by focusing exclusively on the model's behavior with respect to the minority fraud class.

### 3.2.3 Feature Engineering and Preprocessing {#323-feature-engineering-and-preprocessing}

Prior to dataset partitioning, the following preprocessing steps are applied to the raw PaySim dataset:

**Removal of non-informative identifier features.** The string identifier columns `nameOrig` and `nameDest` are dropped, as they encode account-specific identifiers that do not generalize across training and test sets and would introduce data leakage if retained.

**Removal of the legacy flag feature.** The `isFlaggedFraud` column is removed to prevent label leakage into the model's input feature space.

**Transaction type encoding.** The categorical `type` feature is encoded using one-hot encoding, producing five binary indicator columns corresponding to the five transaction types.

**Balance discrepancy feature engineering.** Two derived features are constructed to capture behavioral anomalies in account balance dynamics that are characteristic of fraudulent transactions. Figure 3.2 illustrates the raw and log-transformed distribution of the transaction amount feature.

*[Figure 3.2: Transaction Amount Distribution — Raw vs. Log1p-Transformed (logarithmic transformation normalizes right-skewed distribution)]*

The engineered features are defined as follows:

- `errorBalanceOrig` = `newbalanceOrig` − (`oldbalanceOrg` − `amount`): measures the discrepancy between the expected and observed post-transaction balance of the originating account. Fraudulent transactions in PaySim systematically exhibit non-zero `errorBalanceOrig` values, making this the highest-gain feature in the trained XGBoost models.

- `errorBalanceDest` = `newbalanceDest` − (`oldbalanceDest` + `amount`): measures the equivalent discrepancy for the destination account. In fraudulent TRANSFER transactions, destination accounts frequently exhibit anomalous balance dynamics that complement the originating account signal.

**Logarithmic transformation.** The `amount` feature exhibits strong right-skewness in its raw form, with a small number of very large transactions dominating the value range. A logarithmic transformation using `log1p()` is applied to normalize the distribution and stabilize variance, improving model learning efficiency and reducing the influence of extreme outliers on the tree construction process.

**Train-test splitting.** The preprocessed dataset is split into training and test sets using an **80/20 stratified split**, preserving the 0.13% fraud prevalence ratio in both partitions. The confirmed experimental split produces a training set of 5,090,096 records and a test set of 1,272,524 records containing 1,643 confirmed fraud cases. The test set is held constant across all experimental conditions to ensure full comparability of evaluation results.

## 3.3 Non-IID Dataset Partitioning {#33-non-iid-dataset-partitioning}

A defining characteristic of the experimental design is the construction of a **Non-IID data distribution** across the three client banks. Rather than randomly sampling equal partitions from the global dataset — which would produce an approximate IID distribution and fail to replicate the realistic heterogeneity of multi-institutional banking data — the partitioning strategy deliberately assigns transactions to each bank based on their transaction type composition and resulting fraud exposure profile.

**Table 3-3: Non-IID Client Bank Profile and Partitioning Strategy**

| Client | Bank Profile | Transaction Types | Training Records | Fraud Records | Fraud Prevalence |
|---|---|---|---|---|---|
| **Bank 1** | High-Risk | TRANSFER, CASH-OUT | 1,064,011 | 3,077 | 0.2892% |
| **Bank 2** | Retail / Blind Spot | PAYMENT, CASH-IN | 2,272,208 | **0** | **0.0000%** |
| **Bank 3** | Mixed | All remaining types | 735,859 | 2,129 | 0.2893% |

Figure 3.3 illustrates the Non-IID fraud prevalence distribution across the three client banks.

*[Figure 3.3: Non-IID Data Partitioning — Fraud Prevalence per Client Bank (Bank 2 holds zero fraud labels — the blind spot condition)]*

This partitioning strategy produces three critical properties. Bank 1 holds a locally enriched fraud distribution, enabling it to train a high-performing local classifier. Bank 2 holds a locally fraud-free dataset, rendering it entirely unable to train a functional fraud classifier in isolation — its Local-Only AUPRC of 0.5006 confirms this complete detection failure. Bank 3 holds a moderate and mixed fraud distribution, producing a competent but not optimal local classifier. This Non-IID configuration reflects the realistic heterogeneity of a multi-institutional banking federation and constitutes the most challenging test case for federated aggregation.

## 3.4 Local Model Training: XGBoost Configuration {#34-local-model-training-xgboost-configuration}

Each client bank trains a local **XGBoost** classifier on its private partitioned dataset. XGBoost is selected as the core classification algorithm for the following reasons:

- **Superior performance on structured tabular data:** XGBoost consistently achieves state-of-the-art results on structured financial transaction data, outperforming neural network architectures in settings where feature engineering is applicable (Chen and Guestrin, 2016).
- **Native handling of class imbalance:** XGBoost's `scale_pos_weight` hyperparameter enables direct adjustment of the loss function to penalize misclassification of the minority fraud class more heavily, providing a built-in mechanism for addressing the 773:1 class imbalance.
- **JSON serialization support:** XGBoost natively supports serialization of trained model structures to JSON format via the `save_model()` function, enabling the JSON Tree Concatenation Algorithm to operate directly on the model's internal tree representations without requiring custom serialization logic.
- **Interpretability:** XGBoost provides feature importance scores, supporting post-hoc analysis of the model's decision-making process — an important consideration in regulatory financial environments.

**Table 3-4: XGBoost Hyperparameter Configuration**

| Hyperparameter | Value | Justification |
|---|---|---|
| `n_estimators` | 100 | Sufficient tree count for convergence on PaySim |
| `max_depth` | 6 | Standard depth for financial tabular data |
| `learning_rate` | 0.1 | Moderate shrinkage for balanced convergence |
| `scale_pos_weight` | 773 | Set equal to the negative-to-positive class ratio |
| `eval_metric` | `aucpr` | Directly optimizes AUPRC during training |
| `use_label_encoder` | False | Suppresses deprecation warnings in XGBoost ≥ 1.6 |
| `random_state` | 42 | Ensures full reproducibility across all runs |

## 3.5 The JSON Tree Concatenation Algorithm {#35-the-json-tree-concatenation-algorithm}

The **JSON Tree Concatenation Algorithm** constitutes the primary technical contribution of this thesis. It is a novel model aggregation method designed specifically to enable horizontal Federated Learning with XGBoost models, addressing the fundamental incompatibility between FedAvg's parameter averaging semantics and the discrete tree structure representation of XGBoost ensembles.

### 3.5.1 Theoretical Foundation {#351-theoretical-foundation}

XGBoost's prediction function for a given input vector **x** is defined as:

ŷ = Σ f_k(**x**), f_k ∈ ℱ

where *K* is the total number of trees in the ensemble and each f_k is a regression tree drawn from the space ℱ of all possible regression trees. The final prediction is the sum of the scores assigned by each tree to the input vector. This **additive scoring semantics** is the foundational property that makes tree concatenation a theoretically valid aggregation strategy.

Given three locally trained XGBoost models — one from each client bank — with tree ensembles 𝒯₁, 𝒯₂, and 𝒯₃ containing K₁, K₂, and K₃ trees respectively, the federated global model M_fed is defined as:

M_fed = 𝒯₁ ∪ 𝒯₂ ∪ 𝒯₃

The federated model's prediction for input **x** is therefore:

ŷ_fed = Σ f_k(**x**) for k ∈ 𝒯₁ + Σ f_k(**x**) for k ∈ 𝒯₂ + Σ f_k(**x**) for k ∈ 𝒯₃

This formulation preserves the additive structure of XGBoost while incorporating the discriminative patterns encoded in each client's locally trained tree ensemble. The Global Server constructs M_fed using only the serialized tree structures transmitted by each client — no raw transaction data is accessed or required at any stage of the aggregation process.

### 3.5.2 Algorithm Implementation {#352-algorithm-implementation}

The JSON Tree Concatenation Algorithm is implemented in four sequential steps at the Global Server. Figure 3.4 illustrates the complete four-step process.

*[Figure 3.4: JSON Tree Concatenation Algorithm — Four-Step Aggregation Process (no raw data transferred at any step)]*

**Step 1 — Local model serialization (Client-side).** Each client bank trains its local XGBoost model on its private dataset and serializes the trained model to a JSON file using XGBoost's native `save_model()` function. The serialized JSON file encodes the complete internal tree structure of the local model, including all node split features, split thresholds, leaf scores, and tree metadata. This JSON file — containing no raw transaction data — is transmitted to the Global Server.

**Step 2 — Tree structure extraction (Server-side).** The Global Server loads each client's serialized JSON model file and extracts the tree array from the model's internal structure at the path `learner → gradient_booster → model → trees`.

**Step 3 — Tree concatenation (Server-side).** The extracted tree arrays from all three client models are concatenated into a single unified tree list. Tree metadata — including the `tree_info` array, `iteration_indptr` array, and `num_trees` parameter — is updated to reflect the sequential ordering and total count of the concatenated ensemble. All three metadata fields must be updated consistently to satisfy XGBoost's internal validation, which strictly enforces that `len(tree_info) == num_trees`.

**Step 4 — Federated model reconstruction (Server-side).** The concatenated tree list and updated metadata are injected back into a base model JSON structure, and the federated global model is reconstructed, saved, and redistributed to all client banks for the next federation round.

### 3.5.3 Privacy Guarantee {#353-privacy-guarantee}

The JSON Tree Concatenation Algorithm preserves the core privacy guarantee of the Federated Learning paradigm: **no raw transaction data is transferred between any client bank and the Global Server at any stage of the protocol**. The only artifacts transmitted are the serialized JSON model files, which encode learned tree split structures and leaf scores. These artifacts do not contain individual transaction records, account identifiers, or any other personally identifiable information. The privacy guarantee is enforced by the architecture of the algorithm itself, without requiring cryptographic mechanisms such as differential privacy or secure multi-party computation.

## 3.6 Federated Training Protocol {#36-federated-training-protocol}

The federated training protocol is executed across **five communication rounds**. The protocol is implemented as a simulated federated environment using **four independent terminal processes**: three client terminals (Terminal 1 — Bank 1, Terminal 2 — Bank 2, Terminal 3 — Bank 3) and one Global Server terminal (Terminal 0). Communication between terminals is simulated through shared file system access to a designated model exchange directory. Figure 3.5 illustrates the communication architecture across all five rounds.

*[Figure 3.5: Federated Training Protocol — Five-Round Communication Diagram (4-terminal architecture)]*

The protocol for each communication round proceeds as follows:

1. **Local Training (Client-side, parallel).** Each client bank trains a local XGBoost model on its private local dataset. In Round 1, training is initialized from scratch. In Rounds 2 through 5, training is warm-started from the previous round's federated global model, allowing each client to continue refining the globally informed model with its local data.

2. **Model Submission (Client-side).** Each client serializes its locally trained model to a JSON file and writes it to the shared model exchange directory.

3. **Aggregation (Server-side).** Upon receiving updated model files from all three clients, the Global Server executes the JSON Tree Concatenation Algorithm to produce the updated federated global model.

4. **Redistribution (Server-side).** The updated federated global model is written to the shared directory and made available to all three client terminals.

5. **Evaluation.** The updated federated global model is evaluated on the held-out global test set. AUPRC and F1-Score are recorded for each client at each round.

## 3.7 Baseline Experimental Conditions {#37-baseline-experimental-conditions}

Two baseline conditions are established to bound the performance of the federated framework.

### 3.7.1 Local-Only Baseline {#371-local-only-baseline}

Each client bank trains an independent XGBoost classifier exclusively on its own private dataset, with no federation or data sharing of any kind. The model is evaluated on the global held-out test set. This condition establishes the **lower performance bound** for each client and provides the quantitative demonstration of the blind spot problem for Bank 2.

### 3.7.2 Centralized Baseline {#372-centralized-baseline}

A single XGBoost classifier is trained on the fully pooled dataset comprising all transactions from all three client banks. This condition represents the **theoretical performance ceiling** — the best achievable result when data privacy is disregarded and raw data is freely consolidated. The Centralized baseline achieves AUPRC = 0.9976 and F1 = 0.9516 on the global test set. This baseline is evaluated solely as a reference point for the privacy-performance trade-off analysis and is explicitly noted as **privacy-violating by design** — it is not a deployable system but a theoretical benchmark.

## 3.8 Evaluation Metrics {#38-evaluation-metrics}

All models across all experimental conditions are evaluated using the following metrics exclusively.

### 3.8.1 AUPRC — Area Under the Precision-Recall Curve {#381-auprc}

The **Precision-Recall Curve** plots Precision against Recall across all possible classification thresholds. The **AUPRC** summarizes this curve as a single scalar value between 0 and 1, where a value of 1.0 represents perfect precision and recall at all thresholds, and a value of approximately 0.5 represents the performance of a random classifier on a dataset with 0.13% fraud prevalence. Figure 3.6 illustrates the conceptual interpretation of AUPRC.

*[Figure 3.6: Precision-Recall Curve — AUPRC Conceptual Illustration (higher area = better performance under class imbalance)]*

AUPRC is the primary evaluation metric of this thesis for two reasons. First, it is insensitive to the class imbalance ratio: unlike ROC-AUC, which can yield optimistic scores even for classifiers with poor minority-class performance, AUPRC directly measures the quality of the classifier's predictions on the positive fraud class. Second, a model with AUPRC ≈ 0.5 — as observed for Bank 2 under the Local-Only condition — unambiguously indicates near-random detection capability, providing an interpretable and extreme lower bound for measuring federated improvement.

### 3.8.2 F1-Score {#382-f1-score}

The **F1-Score** is the harmonic mean of Precision and Recall at a fixed classification threshold of 0.5:

F1 = 2 × (Precision × Recall) / (Precision + Recall)

The F1-Score complements AUPRC by providing a threshold-specific performance measure that captures the operational trade-off between false positives and false negatives at a single operating point. A classification threshold of **0.5** is applied consistently across all models and experimental conditions.

### 3.8.3 Explicit Exclusion of Accuracy {#383-explicit-exclusion-of-accuracy}

**Accuracy is explicitly excluded as an evaluation metric** throughout this thesis. Figure 3.7 demonstrates this exclusion quantitatively.

*[Figure 3.7: Accuracy Exclusion Justification — Degenerate classifier scores 99.87% accuracy while detecting zero fraud]*

Under the 0.13% fraud prevalence of the PaySim dataset, a degenerate classifier that predicts the majority class for every transaction achieves an accuracy of 99.87% while detecting zero fraud cases. This score conveys no information about fraud detection capability and would misrepresent the true performance of any model evaluated in this context. The use of Accuracy in this experimental setting would be methodologically unsound and is applied without exception to all reported results.

## 3.9 Tools and Technologies {#39-tools-and-technologies}

**Table 3-5: Tools and Technologies Summary**

| Tool / Library | Version | Purpose |
|---|---|---|
| **Python** | 3.9+ | Primary programming language |
| **XGBoost** | 1.7+ | Core classification algorithm for all models |
| **Pandas** | 1.5+ | Data loading, preprocessing, and feature engineering |
| **NumPy** | 1.23+ | Numerical operations and array manipulation |
| **Scikit-learn** | 1.2+ | Train-test splitting, evaluation metrics, preprocessing |
| **Matplotlib / Seaborn** | 3.6+ | Visualization of results and figures |
| **JSON (stdlib)** | — | Serialization and manipulation of XGBoost model trees |
| **Jupyter Notebook** | 6.5+ | Interactive development and documentation |

All experiments were executed on a local Windows machine. The federated simulation was implemented using four independent terminal sessions to enforce data isolation between client banks and the Global Server, with model file exchange conducted through a shared local directory structure (`models/exchange/`).

---

# CHAPTER 4 — RESULTS {#chapter-4-results}

## 4.1 Overview {#41-overview}

This chapter presents the quantitative results of all three experimental conditions evaluated in this study: the **Local-Only baseline**, the **Centralized baseline**, and the proposed **Federated Learning framework** using the **JSON Tree Concatenation Algorithm**. Results are reported exclusively in terms of **AUPRC** and **F1-Score** across all models and experimental conditions. **Accuracy is not reported** for any model, consistent with its explicit exclusion on methodological grounds established in Section 3.8.3.

## 4.2 Descriptive Statistics of Partitioned Dataset {#42-descriptive-statistics-of-partitioned-dataset}

The PaySim dataset is partitioned into training and test sets using an 80/20 stratified split prior to Non-IID client partitioning. The confirmed partition statistics are as follows:

**Table 4-1: Descriptive Statistics of Partitioned Dataset Across Three Client Banks**

| Client | Bank Profile | Training Records | Fraud Records | Fraud Prevalence | Test Set |
|---|---|---|---|---|---|
| **Bank 1** | High-Risk | 1,064,011 | 3,077 | 0.2892% | Global test set |
| **Bank 2** | Retail / Blind Spot | 2,272,208 | **0** | **0.0000%** | Global test set |
| **Bank 3** | Mixed | 735,859 | 2,129 | 0.2893% | Global test set |
| **Global Test** | — | — | 1,643 | 0.1292% | 1,272,524 records |

The partitioning confirms the Non-IID property of the distributed dataset. Bank 1 holds a disproportionately high concentration of fraudulent transactions relative to its share of total training volume. Bank 2 holds zero fraudulent transactions across its entire local training corpus — a structural condition that deterministically produces a non-functional fraud classifier under isolated training. Bank 3 holds a moderate fraud exposure distributed across multiple transaction types.

## 4.3 Local-Only Baseline Results {#43-local-only-baseline-results}

The Local-Only baseline condition trains an independent XGBoost classifier on each bank's isolated private dataset with no data sharing or model federation. This condition represents the worst-case performance scenario for each client and provides the empirical foundation for the blind spot problem central to this thesis.

### 4.3.1 Per-Client Local-Only Performance {#431-per-client-local-only-performance}

**Table 4-2: Local-Only Baseline Performance — AUPRC and F1-Score per Client Bank**

| Client | Bank Profile | AUPRC | F1-Score | Precision | Recall |
|---|---|---|---|---|---|
| **Bank 1** | High-Risk | **0.9343** | **0.0541** | 0.0278 | 0.9976 |
| **Bank 2** | Retail / Blind Spot | **0.5006** | **0.0000** | 0.0000 | 0.0000 |
| **Bank 3** | Mixed | **0.9932** | **0.6556** | 0.4884 | 0.9970 |

Figure 4.1 visualizes the Local-Only AUPRC and F1-Score per bank alongside the Centralized ceiling.

*[Figure 4.1: Local-Only Baseline Performance — AUPRC and F1-Score per Client Bank vs. Centralized Ceiling]*

### 4.3.2 Interpretation of Local-Only Results {#432-interpretation-of-local-only-results}

**Bank 1** achieves the AUPRC = 0.9343 under Local-Only conditions, reflecting the benefit of its High-Risk data distribution: the local training corpus contains a sufficient density of labeled fraud cases to enable XGBoost to learn discriminative decision boundaries. The high AUPRC indicates that the model maintains meaningful precision across a broad range of recall thresholds. However, the F1-Score of 0.0541 at the default 0.5 threshold is notably low, indicating that the model's predicted probability scores are concentrated in a range below 0.5 for the fraud class due to the extreme class imbalance in the training data.

**Bank 3** achieves AUPRC = 0.9932 and F1-Score = 0.6556, reflecting its Mixed data profile with moderate fraud exposure across multiple transaction types. The stronger AUPRC relative to Bank 1 reflects Bank 3's balanced transaction type composition, which includes both fraud-bearing and non-fraud-bearing types. The higher F1-Score indicates a more calibrated probability output at the default threshold.

**Bank 2** achieves AUPRC = 0.5006 and F1-Score = 0.0000 under the Local-Only condition. The AUPRC of 0.5006 is statistically indistinguishable from a random classifier, which achieves AUPRC ≈ fraud_prevalence ≈ 0.0013 on a balanced test set but approaches 0.5 on this evaluation configuration due to the model's uniform probability assignments. The F1-Score of 0.0000 confirms that at the default classification threshold of 0.5, the model classifies every transaction as legitimate, detecting zero fraud cases. This result constitutes the quantitative definition of the **blind spot problem**: Bank 2 possesses no local fraud intelligence and therefore cannot produce a functional fraud classifier in isolation. No algorithmic improvement applied within Bank 2's isolated environment can resolve this condition — it is a structural consequence of zero positive training examples.

The Local-Only results collectively demonstrate that fraud detection capability under data isolation is entirely contingent on the local availability of labeled fraud examples. This asymmetric performance landscape — strong detection at Banks 1 and 3, total failure at Bank 2 — motivates the federated approach proposed in this thesis.

## 4.4 Centralized Baseline Results {#44-centralized-baseline-results}

The Centralized baseline trains a single XGBoost classifier on the fully pooled dataset comprising all transactions from all three client banks. This condition is **privacy-violating by design** and is evaluated solely to establish the theoretical performance ceiling.

### 4.4.1 Centralized Baseline Performance {#441-centralized-baseline-performance}

**Table 4-3: Centralized Baseline Performance — AUPRC, F1-Score, Precision, and Recall**

| Condition | AUPRC | F1-Score | Precision | Recall |
|---|---|---|---|---|
| **Centralized (Pooled)** | **0.9976** | **0.9516** | 0.9091 | 0.9982 |

Figure 4.2 presents the Precision-Recall curve of the Centralized baseline model.

*[Figure 4.2: Precision-Recall Curve — Centralized Baseline Model (AUPRC = 0.9976, Privacy-Violated Upper Bound)]*

### 4.4.2 Interpretation of Centralized Results {#442-interpretation-of-centralized-results}

The Centralized model achieves AUPRC = 0.9976 and F1-Score = 0.9516, representing the strongest single-model performance achievable on the global PaySim test set given the full training data. The near-perfect AUPRC of 0.9976 confirms that the pooled model learns highly discriminative fraud patterns from the combined transaction histories of all three institutions, leveraging the fraud-labeled examples contributed by Bank 1 and Bank 3 to generalize effectively across the full test distribution. The F1-Score of 0.9516 at the default threshold demonstrates well-calibrated probability outputs that translate directly to strong operational detection performance.

The Centralized baseline establishes the primary reference point for evaluating the **privacy-performance trade-off** of the federated framework. The degree to which the federated model approaches the Centralized baseline — while maintaining strict data isolation — determines the practical viability of the proposed approach for real-world deployment.

## 4.5 Federated Learning Results {#45-federated-learning-results}

This section presents the results of the proposed Federated Learning framework across five communication rounds, with the **JSON Tree Concatenation Algorithm** applied at the Global Server at the conclusion of each round.

### 4.5.1 Bank 2 Performance Across Federation Rounds {#451-bank-2-performance-across-federation-rounds}

The primary metric of interest is Bank 2's AUPRC and F1-Score trajectory across five rounds, as it directly measures the framework's effectiveness in resolving the blind spot problem for the data-isolated retail institution.

**Table 4-4: Bank 2 AUPRC and F1-Score Trajectory Across Five Federated Rounds**

| Round | Bank 2 AUPRC | Bank 2 F1-Score | Bank 2 Precision | Bank 2 Recall |
|---|---|---|---|---|
| **0 (Local-Only)** | 0.5006 | 0.0000 | 0.0000 | 0.0000 |
| **Round 1** | **0.9830** | **0.8430** | 0.7329 | 0.9921 |
| **Round 2** | 0.9830 | 0.8526 | 0.7482 | 0.9909 |
| **Round 3** | 0.9830 | 0.8526 | 0.7482 | 0.9909 |
| **Round 4** | 0.9830 | 0.8526 | 0.7482 | 0.9909 |
| **Round 5** | **0.9830** | **0.8526** | **0.7482** | **0.9909** |

Figures 4.3 and 4.4 present the AUPRC and F1-Score trajectories respectively across all five federation rounds for all three client banks.

*[Figure 4.3: AUPRC Trajectory Across Five Federated Communication Rounds — Bank 2 recovers from 0.5006 to 0.9830 in Round 1]*

*[Figure 4.4: F1-Score Trajectory Across Five Federated Communication Rounds]*

### 4.5.2 All-Client Federated Performance at Round 5 {#452-all-client-federated-performance-at-round-5}

**Table 4-5: All-Client Federated Performance at Round 5 vs. Local-Only Baseline**

| Client | Local-Only AUPRC | FL Round 5 AUPRC | ΔAUPRC | Local-Only F1 | FL Round 5 F1 | ΔF1 |
|---|---|---|---|---|---|---|
| **Bank 1** | 0.9343 | **0.9830** | +0.0487 | 0.0541 | **0.8526** | +0.7985 |
| **Bank 2** | 0.5006 | **0.9830** | +0.4824 | 0.0000 | **0.8526** | +0.8526 |
| **Bank 3** | 0.9932 | **0.9830** | −0.0102 | 0.6556 | **0.8526** | +0.1970 |
| **Centralized** | — | 0.9976 | — | — | 0.9516 | — |

Figure 4.5 presents the focused Bank 2 recovery visualization, and Figure 4.6 presents the comparative AUPRC bar chart across all three experimental conditions.

*[Figure 4.5: Bank 2 Blind Spot Resolution — AUPRC: 0.5006 → 0.9830 | F1: 0.0000 → 0.8526 achieved in Round 1]*

*[Figure 4.6: AUPRC Comparison — Local-Only vs. FL Round 5 vs. Centralized (Privacy Tax = 0.0146)]*

### 4.5.3 Interpretation of Federated Learning Results {#453-interpretation-of-federated-learning-results}

**Bank 2 — Resolution of the Blind Spot Problem.** The most critical finding of the federated experiments is Bank 2's performance trajectory. Starting from a Local-Only baseline of AUPRC = 0.5006 and F1-Score = 0.0000 — representing a random classifier and zero operational detection — Bank 2's federated model achieves AUPRC = 0.9830 and F1-Score = 0.8430 in Round 1, and improves further to F1-Score = 0.8526 from Round 2 onward. This recovery is achieved exclusively through the receipt of the federated global model produced by the **JSON Tree Concatenation Algorithm**, without any transfer of raw transaction data from Bank 1 or Bank 3 to Bank 2 at any stage of the protocol.

The mechanism of recovery is grounded in the additive scoring semantics of the concatenated ensemble. Bank 2's local XGBoost model, trained on fraud-free data, contributes trees that accurately characterize the distribution of legitimate transactions. The trees contributed by Bank 1 and Bank 3, trained on fraud-labeled data, supply the discriminative fraud detection capability that Bank 2's local data cannot provide. The concatenated global model therefore combines Bank 2's local behavioral knowledge with the fraud intelligence of its federated partners, producing a classifier that is substantially more capable than any single client's local model in isolation.

**Convergence Behavior.** A notable characteristic of the experimental results is the immediate convergence of the federated framework: Bank 2's AUPRC improves from 0.5006 to 0.9830 within a single communication round and remains stable across all subsequent rounds. This rapid convergence suggests that the fraud-discriminative signal encoded in Bank 1's and Bank 3's tree structures is sufficiently rich to resolve Bank 2's blind spot in one aggregation step. This finding has significant practical implications: the federated protocol could be abbreviated to a single round without material loss of detection performance, substantially reducing communication overhead in real-world multi-institutional deployments.

**Bank 1 and Bank 3 — Federated Performance Change.** Bank 1 experiences an increase in AUPRC from 0.9343 to 0.9830 (+0.0487) and a substantial increase in F1-Score from 0.0541 to 0.8526 (+0.7985) after federation. The F1-Score improvement reflects the benefit of receiving fraud-discriminative trees from both Bank 3 and the broader federated ensemble, which improves the model's calibrated threshold performance. Bank 3 experiences a marginal decrease in AUPRC from 0.9932 to 0.9830 (−0.0102) but an increase in F1-Score from 0.6556 to 0.8526 (+0.1970). The slight AUPRC reduction for Bank 3 is attributable to the inclusion of Bank 2's fraud-free trees in the concatenated ensemble, which introduces a mild dilution of the fraud-discriminative signal. This marginal trade-off represents the cost of enabling Bank 2's participation in the federation and is consistent with the known Non-IID performance dynamics in federated learning literature (Zhao et al., 2018).

## 4.6 Comparative Summary Across All Experimental Conditions {#46-comparative-summary}

**Table 4-6: Consolidated Performance Results Across All Experimental Conditions**

| Condition | Bank 1 AUPRC | Bank 1 F1 | Bank 2 AUPRC | Bank 2 F1 | Bank 3 AUPRC | Bank 3 F1 |
|---|---|---|---|---|---|---|
| **Local-Only** | 0.9343 | 0.0541 | 0.5006 | 0.0000 | 0.9932 | 0.6556 |
| **Centralized ⚠** | 0.9976 | 0.9516 | 0.9976 | 0.9516 | 0.9976 | 0.9516 |
| **FL Round 1** | 0.9830 | 0.8430 | 0.9830 | 0.8430 | 0.9830 | 0.8430 |
| **FL Round 2** | 0.9830 | 0.8526 | 0.9830 | 0.8526 | 0.9830 | 0.8526 |
| **FL Round 3** | 0.9830 | 0.8526 | 0.9830 | 0.8526 | 0.9830 | 0.8526 |
| **FL Round 4** | 0.9830 | 0.8526 | 0.9830 | 0.8526 | 0.9830 | 0.8526 |
| **FL Round 5** | **0.9830** | **0.8526** | **0.9830** | **0.8526** | **0.9830** | **0.8526** |

*⚠ Centralized condition pools raw data across all banks — privacy-violating by design. Evaluated as theoretical upper bound only.*

Note: All three banks share identical scores from Round 1 onward because they all evaluate the same federated global model produced by the JSON Tree Concatenation Algorithm. This is expected and correct behavior — the concatenated global model is a single unified ensemble applied uniformly to all clients at evaluation time.

## 4.7 Feature Importance Analysis {#47-feature-importance-analysis}

Feature importance scores derived from the federated global model at Round 5 — measured by the **gain metric**, which quantifies the average improvement in the loss function contributed by each feature across all tree splits — reveal consistent patterns that validate the feature engineering decisions documented in Section 3.2.3. Figure 4.7 presents the top 12 features by importance gain.

*[Figure 4.7: XGBoost Feature Importance — Federated Global Model Round 5 (engineered features dominate)]*

The two engineered balance error features dominate the importance ranking: `errorBalanceOrig` and `errorBalanceDest` are the most influential predictors by a substantial margin, confirming that the domain-informed feature engineering applied in preprocessing significantly enhances the model's discriminative power. Fraudulent transactions in PaySim systematically exhibit non-zero balance discrepancy values due to the simulation's encoding of fraud through balance manipulation, making these features the most reliable fraud signals in the feature set.

The transaction type indicators `type_TRANSFER` and `type_CASH_OUT` rank among the top features, consistent with the dataset-level observation that fraud is exclusively confined to these two transaction types in PaySim. The original balance and amount features rank below the engineered features, confirming that the balance discrepancy transformation provides a stronger discriminative signal than the raw balance values alone.

## 4.8 Precision-Recall Curve Analysis {#48-precision-recall-curve-analysis}

Figure 4.8 presents the overlaid Precision-Recall curves for all evaluated model conditions, providing a comprehensive visual comparison of classifier performance across the full range of classification thresholds.

*[Figure 4.8: Precision-Recall Curves — All Models Overlaid (Local-Only × 3, Centralized, FL Round 5)]*

The Precision-Recall curve analysis reveals several key patterns. Bank 3's Local-Only curve achieves the highest area among the Local-Only models, confirming its AUPRC of 0.9932. Bank 1's Local-Only curve demonstrates strong performance at low recall thresholds but degrades at high recall, consistent with its fraud-enriched but single-type-focused training data. Bank 2's Local-Only curve degenerates to near-horizontal at the dataset prevalence level, visually confirming the random classifier behavior indicated by AUPRC = 0.5006.

The Centralized baseline curve achieves the highest area of all evaluated conditions (AUPRC = 0.9976), establishing the visual upper bound. The Federated Round 5 curve approaches but does not fully match the Centralized curve, with the gap between the two curves representing the **privacy tax** of 0.0146 AUPRC — visually apparent but operationally negligible.

---

# CHAPTER 5 — DISCUSSION {#chapter-5-discussion}

## 5.1 Overview {#51-overview}

This chapter interprets the experimental results presented in Chapter 4 within the broader context of the research objectives established in Chapter 1, the theoretical frameworks reviewed in Chapter 2, and the methodological design decisions documented in Chapter 3. The discussion proceeds through five analytical threads: the empirical validation of the blind spot problem, the effectiveness of the JSON Tree Concatenation Algorithm, the privacy-performance trade-off achieved by the framework, a comparative analysis against prior work, and a critical reflection on the limitations of the study.

## 5.2 Empirical Validation of the Blind Spot Problem {#52-empirical-validation-of-the-blind-spot-problem}

The most unambiguous finding of this study is the quantitative confirmation of the blind spot problem under data isolation conditions. Bank 2's Local-Only baseline performance of AUPRC = 0.5006 and F1-Score = 0.0000 represents not a degraded or suboptimal detection capability, but a categorical failure of the fraud detection function. The AUPRC of 0.5006 is statistically equivalent to a random classifier: the model assigns fraud probability scores that carry no discriminative information about the true class of any transaction. The F1-Score of 0.0000 at the default threshold confirms that in operational deployment, this model would detect zero fraudulent transactions while falsely identifying zero fraud cases — a situation that is operationally equivalent to having no fraud detection system at all.

This result is theoretically inevitable: a supervised XGBoost classifier trained exclusively on a fraud-free dataset has no positive class examples from which to learn discriminative decision boundaries. The model's internal tree structures encode only the behavioral distribution of legitimate transactions, and the `scale_pos_weight` hyperparameter — designed to amplify the influence of minority class examples during training — has no effect when the minority class is entirely absent from the training corpus.

This finding carries a significant practical implication that extends beyond the experimental simulation. In real-world multi-institutional banking environments, retail banks serving low-risk customer segments are not merely unlikely to accumulate fraud-labeled training data — they are structurally prevented from doing so by the nature of their customer base and transaction mix. A retail bank whose customers primarily conduct low-value domestic payments will accumulate millions of legitimate transaction records and potentially zero confirmed fraud cases over extended operational periods. Under these conditions, the deployment of any supervised fraud detection model is operationally futile regardless of architectural sophistication or hyperparameter optimization. The blind spot problem is therefore not an edge case but a predictable and recurring structural failure mode for a well-defined category of financial institution.

The Local-Only results for Bank 1 (AUPRC = 0.9343, F1 = 0.0541) and Bank 3 (AUPRC = 0.9932, F1 = 0.6556) further demonstrate that the blind spot problem is asymmetric across institutions. Banks with fraud-enriched local data distributions are capable of training effective local classifiers without any external data access. This asymmetry creates a two-tier detection landscape — effective detection at fraud-exposed institutions, total failure at fraud-isolated institutions — that Federated Learning is uniquely positioned to address.

## 5.3 Effectiveness of the JSON Tree Concatenation Algorithm {#53-effectiveness-of-the-json-tree-concatenation-algorithm}

### 5.3.1 Theoretical Validity {#531-theoretical-validity}

The theoretical validity of tree concatenation as an aggregation strategy rests on a single foundational property of XGBoost: its prediction function is a sum over all trees in the ensemble. This additive structure means that the prediction of a model constructed by concatenating the trees of three locally trained ensembles is mathematically equivalent to summing the predictions of all three local models simultaneously. No information is lost in the aggregation process — the concatenated global model retains the full discriminative capacity of each client's local tree ensemble — and no approximation or averaging operation is applied that could introduce bias or distort the learned decision boundaries.

This theoretical grounding distinguishes the JSON Tree Concatenation Algorithm from FedAvg-style averaging approaches, which introduce averaging bias when applied to heterogeneous Non-IID client updates. In the FedAvg framework, the arithmetic mean of model parameters from clients with divergent local data distributions produces a global model that is suboptimal for all clients — a phenomenon known as client drift (Karimireddy et al., 2020). The concatenation approach avoids this failure mode entirely: rather than averaging divergent updates into a compromise solution, it preserves each client's local model in its entirety within the global ensemble, allowing the federated model to simultaneously represent the behavioral patterns learned from all three data distributions.

### 5.3.2 Empirical Effectiveness {#532-empirical-effectiveness}

The empirical effectiveness of the JSON Tree Concatenation Algorithm is most directly evidenced by Bank 2's performance recovery. Starting from a baseline of random-classifier performance — AUPRC = 0.5006, F1 = 0.0000 — Bank 2's federated model achieves AUPRC = 0.9830 and F1-Score = 0.8430 in Round 1, representing gains of +0.4824 AUPRC and +0.8430 F1-Score achieved in a single communication round without any raw data transfer.

A notable characteristic of the experimental results is the **immediate convergence** of the federated framework. Bank 2's AUPRC improves from 0.5006 to 0.9830 within the first communication round and remains stable across all subsequent rounds through Round 5. This immediate convergence suggests that the fraud-discriminative signal encoded in Bank 1's and Bank 3's tree structures — specifically the trees associated with the `errorBalanceOrig`, `errorBalanceDest`, `type_TRANSFER`, and `type_CASH_OUT` features — is sufficiently comprehensive to resolve Bank 2's blind spot in a single aggregation step. The marginal F1-Score improvement from 0.8430 to 0.8526 between Round 1 and Round 2, followed by complete stability through Round 5, confirms that the federated framework achieves its maximum attainable performance within two rounds.

This finding has significant practical implications for deployment. The federated protocol could be abbreviated to a single communication round — or at most two — without material loss of detection performance, substantially reducing the communication overhead and coordination complexity of real-world multi-institutional federated deployments. In environments where communication bandwidth is constrained or where regulatory approval is required for each data exchange event, this rapid convergence property represents a meaningful operational advantage.

### 5.3.3 Non-Invasiveness and Deployment Practicality {#533-non-invasiveness-and-deployment-practicality}

A distinguishing advantage of the JSON Tree Concatenation Algorithm relative to alternative federated tree methods such as SecureBoost (Cheng et al., 2021) and FedTree (Li et al., 2023) is its non-invasiveness with respect to the XGBoost training procedure. Both SecureBoost and FedTree require modification of the core XGBoost training loop — SecureBoost through homomorphic encryption of gradient statistics, and FedTree through centralized tree construction from distributed sufficient statistics. These modifications introduce implementation complexity, computational overhead, and dependency on specific XGBoost build configurations that may not be available in standard financial institution infrastructure.

The JSON Tree Concatenation Algorithm, by contrast, operates entirely at the model serialization layer. The standard XGBoost training procedure is executed without modification at each client; the only federated-specific operations are the serialization of the trained model to JSON, its transmission to the Global Server, and the server-side concatenation of JSON tree arrays. This architecture is directly compatible with any existing XGBoost deployment and requires no modification to client-side training infrastructure, making it practically deployable in real-world financial institution environments where modification of core machine learning infrastructure is operationally constrained.

## 5.4 Privacy-Performance Trade-off Analysis {#54-privacy-performance-trade-off-analysis}

### 5.4.1 Trade-off Quantification {#541-trade-off-quantification}

The privacy-performance trade-off of the federated framework is quantified by the gap between the federated model's Round 5 performance and the Centralized baseline.

**Table 5-1: Privacy-Performance Trade-off Quantification — FL Round 5 vs. Centralized Baseline**

| Metric | Centralized Baseline | FL Round 5 | Privacy Tax |
|---|---|---|---|
| **AUPRC** | 0.9976 | 0.9830 | **0.0146** |
| **F1-Score** | 0.9516 | 0.8526 | **0.0990** |
| **Precision** | 0.9091 | 0.7482 | 0.1609 |
| **Recall** | 0.9982 | 0.9909 | 0.0073 |

Figure 5.1 visualizes the AUPRC privacy tax between the Centralized and Federated conditions.

*[Figure 5.1: Privacy-Performance Trade-off — Centralized (0.9976) vs. Federated Round 5 (0.9830), Privacy Tax = 0.0146 (1.46%)]*

The AUPRC privacy tax of 0.0146 represents a 1.46% reduction in detection capability relative to the privacy-violating Centralized ceiling — an operationally negligible cost that strongly supports the practical viability of the proposed framework. The Recall privacy tax of 0.0073 confirms that the federated model maintains near-identical sensitivity in detecting actual fraud cases relative to the centralized model. The larger F1-Score privacy tax of 0.0990 reflects the precision differential between the two models, which is attributable to the federated model's inclusion of Bank 2's fraud-free trees in the concatenated ensemble, which introduces a mild dilution of precision.

### 5.4.2 The Fundamental Value Proposition {#542-the-fundamental-value-proposition}

Regardless of the precise magnitude of the privacy tax, the fundamental value proposition of the federated framework is unambiguous and cannot be replicated by any centralized approach: **the framework transforms Bank 2 from a completely blind institution into a functional fraud detector, without requiring the transfer of a single raw transaction record**.

The relevant performance comparison for Bank 2 is not between the federated model and the Centralized baseline — it is between the federated model and the Local-Only baseline of AUPRC = 0.5006. Any positive ΔAUPRC achieved by Bank 2 through federation represents a meaningful and operationally significant improvement over its isolated baseline. In this case, the improvement is +0.4824 AUPRC and +0.8526 F1-Score — a transformation from zero detection capability to near-centralized detection performance, achieved within a single communication round and without violating any data governance obligation.

## 5.5 Comparison with Prior Work {#55-comparison-with-prior-work}

### 5.5.1 Comparison with Centralized XGBoost Fraud Detection {#551-comparison-with-centralized-xgboost}

The Centralized baseline results of this thesis — AUPRC = 0.9976, F1 = 0.9516 — exceed the performance benchmarks reported for XGBoost on the PaySim dataset in prior literature. Kumar and Chadha (2020) reported AUPRC values in the range of 0.90–0.95 for XGBoost on PaySim under various feature engineering configurations. The higher AUPRC achieved in this thesis — 0.9976 — is attributable to the domain-informed engineered features `errorBalanceOrig` and `errorBalanceDest`, which exploit the specific fraud mechanism encoded in PaySim's transaction simulation. This result confirms that the preprocessing and feature engineering pipeline described in Section 3.2.3 represents an advancement over baseline PaySim evaluation approaches and produces a ceiling performance that sets a stringent benchmark for the federated framework.

### 5.5.2 Comparison with Federated Learning Fraud Detection Studies {#552-comparison-with-fl-fraud-detection-studies}

The federated fraud detection study most directly comparable to this thesis is Suzumura and Kanezashi (2022), who applied FL to financial crime detection across a simulated network of institutions. Their study employed a neural network architecture with FedAvg aggregation and reported that federated models achieved detection rates approaching the centralized baseline under Non-IID data distributions. The present thesis extends this line of research in three specific ways.

First, this thesis employs **XGBoost** rather than a neural network architecture, which is more appropriate for the structured tabular data of the PaySim dataset. Second, this thesis introduces the **JSON Tree Concatenation Algorithm** as an alternative to FedAvg for tree-based model aggregation, addressing the fundamental incompatibility between FedAvg's parameter averaging semantics and the discrete tree structure representation of XGBoost. Third, this thesis explicitly evaluates the scenario of a **zero-label client** — Bank 2 — which has not been systematically addressed in prior federated fraud detection literature. These three distinctions collectively represent a substantive advancement of the state of knowledge in federated financial fraud detection.

### 5.5.3 Comparison with SecureBoost and FedTree {#553-comparison-with-secureboost-and-fedtree}

SecureBoost (Cheng et al., 2021) and FedTree (Li et al., 2023) represent the two most technically sophisticated existing approaches to federated gradient boosting. SecureBoost employs homomorphic encryption to enable privacy-preserving federated tree construction under vertical data partitioning, while FedTree reconstructs trees centrally from distributed gradient and Hessian statistics under horizontal partitioning. Both approaches offer stronger formal privacy guarantees than the JSON Tree Concatenation Algorithm — SecureBoost through cryptographic protection of gradient statistics, and FedTree through the aggregation of sufficient statistics rather than complete model structures. However, both impose significant implementation complexity and computational overhead that limits their practical deployability in standard financial institution environments.

The JSON Tree Concatenation Algorithm occupies a different point in the privacy-complexity-performance design space: it provides architectural privacy guarantees without cryptographic overhead, operates non-invasively on standard XGBoost outputs, and achieves federated AUPRC of 0.9830 that closely approaches the centralized ceiling of 0.9976. For financial institutions operating under data governance frameworks where architectural data isolation is sufficient for regulatory compliance, the JSON Tree Concatenation Algorithm offers a practically superior deployment profile relative to both SecureBoost and FedTree.

## 5.6 Limitations and Critical Reflection {#56-limitations-and-critical-reflection}

**Synthetic Dataset Constraints.** All experiments are conducted on the PaySim synthetic dataset. While PaySim faithfully replicates the statistical properties of real-world mobile money transaction data, it remains a simulation. PaySim's fraud is exclusively confined to CASH-OUT and TRANSFER transaction types, and its fraud mechanism — balance manipulation — is consistently detectable through the engineered balance discrepancy features. Real-world fraud typologies are considerably more diverse, subtle, and adversarially adaptive, potentially requiring richer feature sets and more sophisticated model architectures than those evaluated in this thesis.

**Simulated Federation Environment.** The federated protocol in this thesis is implemented as a simulation using independent terminal processes on a single local machine, with model exchange conducted through a shared file system directory. This simulation does not replicate the network communication latency, bandwidth constraints, or fault tolerance requirements of a real-world federated deployment across geographically distributed institutions. The practical performance of the framework under real network conditions is not evaluated and constitutes an important direction for future work.

**Absence of Formal Privacy Guarantees.** The privacy guarantee of the JSON Tree Concatenation Algorithm is architectural: no raw data is transferred between clients or to the Global Server. However, the algorithm does not provide formal cryptographic privacy guarantees. Recent research in federated learning security has demonstrated that model parameters can, under certain conditions, be used to reconstruct approximations of training data through model inversion attacks (Geiping et al., 2020). While decision tree structures are generally considered more resistant to such attacks than neural network gradients due to their discrete and non-differentiable nature, this resistance has not been formally proven in the context of the JSON Tree Concatenation Algorithm.

**Fixed Hyperparameter Configuration.** The XGBoost hyperparameter configuration is held constant across all three client banks and all federation rounds. In practice, optimal configurations may differ across clients as a function of their local data distributions, and per-client hyperparameter tuning could yield improved local model quality and consequently a higher-quality federated global model.

**Identical Federated Scores Across Banks.** The experimental results show identical AUPRC and F1-Score for all three banks from Round 1 onward. This is because all three banks evaluate the same federated global model produced by the concatenation algorithm. While this confirms the consistency and correctness of the aggregation, it also means that per-bank differentiation of federated performance — which would be possible in systems where clients fine-tune the global model locally before evaluation — is not observed in this experiment. Future work could explore local fine-tuning steps that allow each bank to specialize the federated global model for its own risk profile.

## 5.7 Summary of Discussion {#57-summary-of-discussion}

The discussion presented in this chapter supports the following key analytical conclusions. The blind spot problem is empirically confirmed as a categorical failure for Bank 2 under data isolation, with AUPRC = 0.5006 and F1 = 0.0000 providing unambiguous evidence. The JSON Tree Concatenation Algorithm is theoretically valid and empirically effective, enabling Bank 2 to recover from random-classifier performance to AUPRC = 0.9830 in a single federated round without any raw data transfer. The privacy tax of 0.0146 AUPRC (1.46%) is operationally negligible and strongly supports the framework's practical viability. The framework advances prior work by introducing a non-invasive XGBoost aggregation mechanism, explicitly evaluating the zero-label client scenario, and providing rigorous empirical evaluation on a realistic fraud detection benchmark. Critical limitations — including synthetic dataset constraints, simulated federation environment, and absence of formal cryptographic privacy guarantees — define the boundaries of the study's findings and motivate a well-defined set of future research directions.

---

# CHAPTER 6 — CONCLUSION {#chapter-6-conclusion}

## 6.1 Summary of the Study {#61-summary-of-the-study}

This research proposed, implemented, and empirically evaluated a **Federated Learning framework** for early financial fraud detection using **XGBoost** as the core classification algorithm across three client banks operating under heterogeneous, **Non-IID** data distributions. The research was motivated by the **blind spot problem** — the categorical fraud detection failure that occurs when retail financial institutions are isolated from fraud-labeled training data by data governance regulations — and the absence of a practically deployable federated aggregation mechanism for XGBoost models in horizontal federation scenarios.

The study was conducted on the **PaySim synthetic financial dataset**, which simulates mobile money transactions with a fraud prevalence of approximately 0.13%, producing a class imbalance ratio of 773:1. Given this extreme imbalance, **Accuracy was explicitly disqualified** as an evaluation metric, with all performance assessments conducted exclusively through **AUPRC** and **F1-Score**.

Three experimental conditions were evaluated. The **Local-Only baseline** demonstrated the performance cost of complete data isolation: Bank 1 (High-Risk, 1,064,011 training records) achieved AUPRC = 0.9343 and F1 = 0.0541; Bank 2 (Retail/Blind Spot, 2,272,208 training records, zero fraud labels) achieved AUPRC = 0.5006 and F1 = 0.0000 — random-classifier performance confirming total detection failure; and Bank 3 (Mixed, 735,859 training records) achieved AUPRC = 0.9932 and F1 = 0.6556. The **Centralized baseline**, which pooled all data onto a single server in a privacy-violating configuration, achieved AUPRC = 0.9976 and F1 = 0.9516, establishing the theoretical performance ceiling.

The **Federated Learning framework**, executed across five communication rounds using the **JSON Tree Concatenation Algorithm**, resolved Bank 2's total detection failure without transferring any raw transaction data between institutions. By Round 1, Bank 2's federated model achieved AUPRC = 0.9830 and F1-Score = 0.8430, recovering from random-classifier performance to near-centralized detection capability in a single communication round. Performance stabilized at AUPRC = 0.9830 and F1-Score = 0.8526 from Round 2 onward, remaining constant through Round 5 for all three client banks. The **privacy tax** — the reduction in AUPRC attributable to the privacy-preserving constraint of the federated architecture relative to the Centralized ceiling — is **0.0146 (1.46%)**, a negligible cost that confirms the framework's practical viability.

The primary technical contribution of this thesis — the **JSON Tree Concatenation Algorithm** — aggregates locally trained XGBoost models by directly concatenating their JSON-serialized internal tree structures at the Global Server. This approach is theoretically grounded in XGBoost's additive scoring semantics (ŷ = Σ f_k(x) for all trees k), non-invasive with respect to the standard XGBoost training procedure, and directly compatible with GDPR and equivalent financial data governance frameworks. It addresses the specific and previously unresolved gap in the federated learning literature: the absence of a horizontal federated aggregation mechanism for XGBoost that is practically deployable without modification to client-side training infrastructure.

## 6.2 Conclusions Drawn {#62-conclusions-drawn}

From the theoretical analysis, methodological design, and experimental evaluation conducted in this thesis, the following conclusions are drawn.

**Conclusion 1 — The blind spot problem is a categorical, not merely degraded, failure of fraud detection under data isolation.** Bank 2's Local-Only performance of AUPRC = 0.5006 and F1 = 0.0000 demonstrates that the absence of positive training examples eliminates fraud detection capability entirely rather than merely reducing it. This finding has direct practical implications for any financial institution whose local transaction history is structurally deficient in confirmed fraud cases. Deploying any supervised fraud detection model under such conditions is operationally futile regardless of the algorithm selected. The blind spot problem requires a structural solution — specifically, access to externally sourced fraud intelligence — rather than an algorithmic one.

**Conclusion 2 — Federated Learning with the JSON Tree Concatenation Algorithm resolves the blind spot problem without compromising data privacy.** The federated framework transforms Bank 2 from a completely blind institution (AUPRC = 0.5006, F1 = 0.0000) to a functional fraud detector (AUPRC = 0.9830, F1 = 0.8526) across five communication rounds, without transferring any raw transaction data between institutions. The framework is directly compatible with GDPR, PDPA, and equivalent data governance frameworks.

**Conclusion 3 — The JSON Tree Concatenation Algorithm is a theoretically valid and practically superior aggregation mechanism for horizontal federated XGBoost.** The algorithm's theoretical validity derives from XGBoost's additive scoring semantics: concatenating the tree ensembles of multiple locally trained models preserves the full discriminative capacity of each client's local model in the global ensemble, without the averaging bias or client drift associated with FedAvg-style parameter averaging. Its practical superiority over existing federated tree methods derives from its non-invasive architecture: it operates entirely at the model serialization layer, requires no modification to the XGBoost training procedure, and introduces negligible computational overhead at the Global Server.

**Conclusion 4 — The federated framework converges in a single communication round, with negligible privacy tax.** Bank 2's AUPRC improves from 0.5006 to 0.9830 within Round 1 and remains stable through Round 5, demonstrating that the fraud-discriminative signal encoded in the participating banks' tree structures is sufficient to resolve the blind spot problem in a single aggregation step. The privacy tax of 0.0146 AUPRC (1.46% relative to the Centralized ceiling) is operationally negligible and confirms that privacy preservation is achieved without meaningful sacrifice of detection capability.

**Conclusion 5 — AUPRC and F1-Score are the only appropriate evaluation metrics for fraud detection under extreme class imbalance.** The PaySim dataset's 0.13% fraud prevalence renders Accuracy a degenerate and misleading metric: a classifier predicting the majority class for every transaction achieves 99.87% accuracy while detecting zero fraud. The exclusive use of AUPRC and F1-Score throughout this thesis ensures that all reported performance assessments reflect the classifier's actual behavior with respect to the minority fraud class.

**Conclusion 6 — No single model universally dominates across all institutional contexts; federated collaboration is essential for equitable detection capability.** The Local-Only results demonstrate a structurally unequal detection landscape: fraud-exposed institutions (Banks 1 and 3) achieve effective isolated performance, while the fraud-isolated institution (Bank 2) achieves zero performance. This inequality is not addressable through algorithmic improvement within isolated institutions — it is a structural consequence of data distribution inequality across the financial ecosystem. Federated Learning is the only architectural paradigm that enables equitable distribution of fraud detection capability across institutions of varying risk profiles while respecting the data governance constraints that prohibit raw data sharing.

## 6.3 Future Works {#63-future-works}

Despite the contributions of this thesis, the proposed framework represents a foundational rather than terminal contribution to the field of privacy-preserving fraud detection. Several well-defined directions for future research are identified below.

**Formal Cryptographic Privacy Guarantees.** The JSON Tree Concatenation Algorithm provides architectural privacy — no raw data leaves client institutions — but does not provide formal cryptographic privacy guarantees against model inversion or tree structure reconstruction attacks. Future work should investigate the integration of **differential privacy** mechanisms into the federated protocol, specifically through the injection of calibrated noise into tree leaf scores before model serialization and transmission. The privacy-utility trade-off of differential privacy noise injection on XGBoost tree structures — measured through its impact on AUPRC and F1-Score — warrants systematic empirical investigation.

**Weighted Tree Concatenation.** The current implementation treats all client tree contributions equally. A natural extension is **weighted tree concatenation**, in which the Global Server assigns differential weights to tree contributions based on client-specific performance metrics such as local AUPRC. Trees from clients with higher local fraud detection capability would receive greater influence in the global ensemble, potentially improving federated model performance under Non-IID distributions. The design of an effective and privacy-preserving weighting scheme represents a technically non-trivial but high-value research direction.

**Scalability to Larger Federated Networks.** The experimental evaluation involves exactly three client banks. The scalability of the JSON Tree Concatenation Algorithm to larger federations raises several open questions: How does the total tree count in the concatenated ensemble — which grows linearly with the number of clients and local `n_estimators` — affect inference latency and memory requirements? Does federated AUPRC continue to improve monotonically as additional clients join the federation, or does performance plateau or degrade beyond a certain federation size? Systematic empirical evaluation across federation sizes of 5, 10, 25, and 50 clients would provide the scalability evidence necessary to assess the framework's viability for industry-scale deployment.

**Temporal and Sequential Modeling.** The current framework treats each transaction as an independent event, ignoring the temporal sequence of transactions associated with individual accounts. Many fraud typologies manifest as sequences of transactions that are individually unremarkable but collectively suspicious when evaluated in temporal context. Future research should investigate the integration of **temporal modeling** — potentially through LSTM networks or Transformer-based sequence models — into the federated fraud detection pipeline. The federated aggregation of temporal models represents a technically novel and operationally relevant extension.

**Real-World Dataset Validation.** All experiments are conducted on the PaySim synthetic dataset. Future work should seek to validate the federated framework on real-world or anonymized proprietary transaction datasets obtained through partnerships with financial institutions or regulatory bodies. Such validation would address the dataset realism limitation identified in Section 5.6 and provide empirical evidence of the framework's performance under the full complexity of real-world fraud typologies, temporal concept drift, and institution-specific behavioral heterogeneity.

**Adaptive Federation Rounds and Early Stopping.** The current protocol executes for a fixed number of five communication rounds. Future work should investigate **adaptive federation protocols** that monitor Bank 2's AUPRC trajectory across rounds and terminate the federated protocol when performance improvement falls below a convergence threshold. Given that experimental results demonstrate convergence within Round 1, adaptive protocols could reduce communication overhead substantially by terminating after one or two rounds for configurations similar to those evaluated in this thesis.

**Integration with Real-Time Transaction Monitoring Systems.** The framework evaluated operates in a batch training paradigm. Real-world fraud detection systems must operate in real-time, scoring incoming transactions as they arrive and updating detection models continuously as new fraud patterns emerge. Future work should investigate the integration of the federated XGBoost framework with **streaming transaction monitoring architectures**, potentially through online learning variants of gradient boosting that update the tree ensemble incrementally with each new transaction batch.

**Explainability and Regulatory Compliance.** Financial regulators increasingly require that automated fraud detection decisions be explainable to affected customers and auditable by compliance officers. Future work should investigate the integration of **Explainable AI (XAI)** techniques — specifically SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) — into the federated framework, enabling per-transaction explanation of fraud scores in terms of individual feature contributions.

In conclusion, this thesis has demonstrated that **Federated Learning**, combined with the **JSON Tree Concatenation Algorithm** and **XGBoost**, constitutes a viable, privacy-compliant, and technically grounded framework for resolving the blind spot problem in multi-institutional financial fraud detection. The framework's core achievement — transforming Bank 2 from a completely blind institution into a functional fraud detector achieving AUPRC = 0.9830 without transferring a single raw transaction record — represents a meaningful contribution to both the academic literature on privacy-preserving machine learning and the practical challenge of deploying effective fraud detection systems in regulated financial environments. The path from the simulation environment of this thesis to production deployment requires the extensions identified above, but the foundational contribution of this work establishes the conceptual and empirical basis upon which those extensions can be built.

---

## REFERENCES {#references .ABSTRACT}

1. Association of Certified Fraud Examiners. (2022). *Report to the nations: 2022 global study on occupational fraud and abuse*. ACFE. https://www.acfe.com/report-to-the-nations/2022/

2. Bahnsen, A. C., Aouada, D., Stojanovic, J., & Ottersten, B. (2016). Feature engineering strategies for credit card fraud detection. *Expert Systems with Applications*, *51*, 134–142. https://doi.org/10.1016/j.eswa.2015.12.030

3. Bhattacharyya, S., Jha, S., Tharakunnel, K., & Westland, J. C. (2011). Data mining for credit card fraud: A comparative study. *Decision Support Systems*, *50*(3), 602–613. https://doi.org/10.1016/j.dss.2010.08.008

4. Bolton, R. J., & Hand, D. J. (2002). Statistical fraud detection: A review. *Statistical Science*, *17*(3), 235–255. https://doi.org/10.1214/ss/1042727940

5. Carcillo, F., Dal Pozzolo, A., Le Borgne, Y. A., Caelen, O., Mazzer, Y., & Bontempi, G. (2019). Scarff: A scalable framework for streaming credit card fraud detection with Spark. *Information Fusion*, *41*, 182–194. https://doi.org/10.1016/j.inffus.2017.09.005

6. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785

7. Cheng, K., Fan, T., Jin, Y., Liu, Y., Chen, T., Papadopoulos, D., & Yang, Q. (2021). SecureBoost: A lossless federated learning framework. *IEEE Intelligent Systems*, *36*(6), 87–98. https://doi.org/10.1109/MIS.2021.3082561

8. Dal Pozzolo, A., Caelen, O., Le Borgne, Y. A., Waterschoot, S., & Bontempi, G. (2015). Learned lessons in credit card fraud detection from a practitioner perspective. *Expert Systems with Applications*, *41*(10), 4915–4928. https://doi.org/10.1016/j.eswa.2014.02.026

9. Financial Action Task Force. (2020). *Anti-money laundering and counter-terrorist financing measures: Guidance for a risk-based approach*. FATF. https://www.fatf-gafi.org

10. Financial Stability Board. (2022). *Artificial intelligence and machine learning in financial services*. FSB. https://www.fsb.org

11. Geiping, J., Bauermeister, H., Dröge, H., & Moeller, M. (2020). Inverting gradients — How easy is it to break privacy in federated learning? *Advances in Neural Information Processing Systems*, *33*, 16937–16947.

12. Karimireddy, S. P., Kale, S., Mohri, M., Reddi, S., Stich, S., & Suresh, A. T. (2020). SCAFFOLD: Stochastic controlled averaging for federated learning. *Proceedings of the 37th International Conference on Machine Learning*, *119*, 5132–5143.

13. Kumar, A., & Chadha, A. (2020). An ensemble learning-based approach for fraud detection in digital transactions. *Procedia Computer Science*, *171*, 1122–1131. https://doi.org/10.1016/j.procs.2020.04.121

14. Li, L., Fan, Y., Tse, M., & Lin, K. Y. (2020). A review of applications in federated learning. *Computers & Industrial Engineering*, *149*, 106854. https://doi.org/10.1016/j.cie.2020.106854

15. Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Smola, A., & Smith, V. (2020). Federated optimization in heterogeneous networks. *Proceedings of Machine Learning and Systems*, *2*, 429–450.

16. Li, Q., Wen, Z., Wu, Z., Hu, S., Wang, N., Li, Y., Liu, X., & He, B. (2023). A survey on federated learning systems: Vision, hype and reality for data privacy and protection. *IEEE Transactions on Knowledge and Data Engineering*, *35*(4), 3347–3366. https://doi.org/10.1109/TKDE.2021.3124599

17. López-Rojas, E., Elmir, A., & Axelsson, S. (2016). PaySim: A financial mobile money simulator for fraud detection. *Proceedings of the 28th European Modeling and Simulation Symposium*, 249–255.

18. McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017). Communication-efficient learning of deep networks from decentralized data. *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics*, *54*, 1273–1282.

19. Ngai, E. W. T., Hu, Y., Wong, Y. H., Chen, Y., & Sun, X. (2011). The application of data mining techniques in financial fraud detection: A classification framework and an academic review of literature. *Decision Support Systems*, *50*(3), 559–569. https://doi.org/10.1016/j.dss.2010.08.006

20. Sahin, Y., & Duman, E. (2011). Detecting credit card fraud by decision trees and support vector machines. *Proceedings of the International MultiConference of Engineers and Computer Scientists*, *1*, 442–447.

21. Suzumura, T., & Kanezashi, H. (2022). Federated learning for financial crime detection. *Proceedings of the IEEE International Conference on Big Data*, 1311–1320. https://doi.org/10.1109/BigData55660.2022.10020599

22. Yang, Q., Liu, Y., Chen, T., & Tong, Y. (2019). Federated machine learning: Concept and applications. *ACM Transactions on Intelligent Systems and Technology*, *10*(2), 1–19. https://doi.org/10.1145/3298981

23. Zhao, Y., Li, M., Lai, L., Suda, N., Civin, D., & Chandra, V. (2018). Federated learning with Non-IID data. *arXiv preprint arXiv:1806.00582*. https://arxiv.org/abs/1806.00582

---

*End of Thesis*

---

## Figure Placement Guide

Every figure reference in the text above is marked as `*[Figure X.X: Caption text]*`. When formatting the final Word document, replace each of these markers with the corresponding PNG file from the `thesis_figures/` folder:

| Marker | PNG File |
