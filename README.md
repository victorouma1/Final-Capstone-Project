# Automated Risk Classification of Aviation Safety Narratives

### Project Overview
This capstone project focuses on building an end-to-end NLP pipeline that automatically processes unstructured aviation safety narratives, predicts the primary Anomaly Category (e.g., Inflight Event, Equipment/Tooling, Communication Breakdown)

### Problem Statement
The aviation industry relies heavily on Voluntary Safety Reporting Programs (VSRPs), such as NASA's Aviation Safety Reporting System (ASRS), to catch latent system vulnerabilities before they escalate into catastrophic accident. Every month, thousands of unstructured, free-text narratives are submitted by pilots, air traffic controllers, and mechanics describing anomalies, near-misses, and mechanical failures. Currently, evaluating these reports is highly labor-intensive. This creates a lag due to the manual processing an sorting means critical, emerging safety risks might linger unnoticed in databases for weeks

### Dataset Description
Source: https://asrs.arc.nasa.gov/search/database.html <br>
Size: 30,000 observations, 125 columns <br>
Key Features: Narrative, Anomaly type <br>

### Tools Used:
Language: Python 3.13 <br>
Data Manipulation: Pandas, NumPy <br>
Data Visualization: Matplotlib, Seaborn <br>
Machine learning:Scikit-learn, keras
Models: <br>
TF-IDF + Baseline Classification Model (e.g. SVM) <br>
BERT <br>
RoBERTa <br>
AeroBERT / Aviation-BERT <br>
Deployment : Streamlit

### Methodology
#### 1. Data Cleaning
Isolated target variable and narrative then dropped the rest of the columns<br>
Changed column name for readability<br>
Droppped missing values they were less than 1% of dataset<br>

#### 2. Data Preprocessing & Feature Engineering
1. Made Narrative Lowercase<br>
2. Removed punctuation<br>
3. Removed spelling mistakes<br>
4. Removed Stopwords <br>
5. Dealt with excessive whitspaces <br>
6. Tokenization<br>
7. Lemmitisation<br>

#### 3. Modeling & Evaluation
Use confusion matrix<br>
Main Evaluation metrics: Recall, Precison, F1-score<br>
Error Analysis to see where model goes wrong<br>

### Authors & Acknowledgments
Victor Ouma - Data Scientist <br>
Thanks to Zindua School for guidance during this capstone project.






