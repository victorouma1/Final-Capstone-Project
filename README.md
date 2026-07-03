# Automated Risk Classification of Aviation Safety Narratives

### Project Overview
This capstone project focuses on building an end-to-end NLP pipeline that automatically processes unstructured aviation safety narratives, predicts the primary Anomaly Category (e.g., Inflight Event, Equipment/Tooling, Communication Breakdown)

### Problem Statement
The aviation industry relies heavily on Voluntary Safety Reporting Programs (VSRPs), such as NASA's Aviation Safety Reporting System (ASRS), to catch latent system vulnerabilities before they escalate into catastrophic accident. Every month, thousands of unstructured, free-text narratives are submitted by pilots, air traffic controllers, and mechanics describing anomalies, near-misses, and mechanical failures. Currently, evaluating these reports is highly labor-intensive. This creates a lag due to the manual processing an sorting means critical, emerging safety risks might linger unnoticed in databases for weeks

### Dataset Description
Source: https://asrs.arc.nasa.gov/search/database.html <br>
Size: 30,000 observations, 125 columns <br>
Key Features: Narrative, Anomaly type <br>
<br>
Target Variable Catgeories and their value counts:<br>
|Issue Catgeory                                     |Value count|
|:--------------:                                   |:---------:|
|Human Factors                                      |10823      |
|Aircraft                                           |9802       |
|Procedure                                          |2913       |
|Ambiguous                                          |2490       |
|Weather                                            |1059       |
|Environment - Non Weather Related                  |930        |
|Airport                                            |786        |
|Chart Or Publication                               |447        |
|Airspace Structure                                 |411        |
|ATC Equipment / Nav Facility / Buildings           |410        |
|Company Policy                                     |408        |
|Software and Automation                            |177        |
|Equipment / Tooling                                |103        |
|Staffing                                           |96         |
|MEL                                                |69         |
|Incorrect / Not Installed / Unavailable Part       |55         |
|Manuals                                            |48         |
|Logbook Entry                                      |11         |
|Primary Problem                                    |8          |
|Human Factors; Aircraft                            |4          |
|Procedure; Aircraft                                |3          |
|Weather; Human Factors                             |2          |
|Company Policy; Aircraft                           |1          |
|Aircraft; Aircraft                                 |1          |
|Ambiguous; Human Factors                           |1          |
|Human Factors; Weather                             |1          |
|Human Factors; Procedure                           |1          |
|Chart Or Publication; Aircraft                     |1          |
|ATC Equipment / Nav Facility / Buildings; Aircraft |1          |
|Aircraft; Airspace Structure                       |1          |


### Tools Used:
Language: Python 3.13 <br>
Data Manipulation: Pandas, NumPy <br>
Data Visualization: Matplotlib, Seaborn <br>
Machine learning:Scikit-learn, keras
Models: <br>
TF-IDF + Baseline Classification Model (e.g. SVM) <br>
DistillBERT <br>
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
8. Use contextual embeddings to capture deep contextual meanings

### 3. How Models work
***BERT***<br>
![alt text](<Image Files/image.png>)<br>
BERT changed the way machines interpret human language. Short for Bidirectional Encoder Representations from Transformers, it allows models to understand context by reading text in both directions

BERT relies on a Transformer (the attention mechanism that learns contextual relationships between words in a text). A basic Transformer consists of an encoder to read the text input and a decoder to produce a prediction for the task.

1. Token embeddings: A [CLS] token is added to the input word tokens at the beginning of the first sentence and a [SEP] token is inserted at the end of each sentence.<br>
2. Segment embeddings: A marker indicating Sentence A or Sentence B is added to each token. This allows the encoder to distinguish between sentences.<br>
3. Positional embeddings: A positional embedding is added to each token to indicate its position in the sentence.

***ROBERTA***<br>
RoBERTa (Robustly Optimized BERT Approach) is a state-of-the-art language representation model developed by Facebook AI. It is based on the original BERT (Bidirectional Encoder Representations from Transformers) architecture but differs in several key ways.

One of the ways is dynamic masking which involves randomly masking different tokens at different points during pre-training as compared to static masking use by BERT which uses the same mask every time

In the original BERT model, the pre-training phase includes a next-sentence prediction (NSP) task, where the model is trained to predict whether a given sentence is the next sentence in a text or not.In RoBERTa, this NSP loss is not used during pre-training. RoBERTa is able to learn a more reliable representation of the language by training the model on complete sentences as opposed to sentence pairs.

***AEROBERT***<br>
A BERT model which has been pretrained on aeronautical terms

#### 4. Modeling & Evaluation
Main Evaluation metrics: F1-score<br>
Model Explainability<br>

### 5. Error Analysis
Confusion matrix <br>
Most confident errors <br>
Most influential tokens <br>

### Deployed model
https://aerobert-safety-report-classifier.streamlit.app/


### Authors & Acknowledgments
Victor Ouma - Data Scientist <br>
Thanks to Zindua School for guidance during this capstone project.






