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
|Issue Catgeory                                |Value count|
|:--------------:                              |:---------:|
|Human Factors                                 |10823      |
Aircraft                                               9802<br>
Procedure                                              2913<br>
Ambiguous                                              2490<br>
Weather                                                1059<br>
Environment - Non Weather Related                       930<br>
Airport                                                 786<br>
Chart Or Publication                                    447<br>
Airspace Structure                                      411<br>
ATC Equipment / Nav Facility / Buildings                410<br>
Company Policy                                          408<br>
Software and Automation                                 177<br>
Equipment / Tooling                                     103<br>
Staffing                                                 96<br>
MEL                                                      69<br>
Incorrect / Not Installed / Unavailable Part             55<br>
Manuals                                                  48<br>
Logbook Entry                                            11<br>
Primary Problem                                           8<br>
Human Factors; Aircraft                                   4<br>
Procedure; Aircraft                                       3<br>
Weather; Human Factors                                    2<br>
Company Policy; Aircraft                                  1<br>
Aircraft; Aircraft                                        1<br>
Ambiguous; Human Factors                                  1<br>
Human Factors; Weather                                    1<br>
Human Factors; Procedure                                  1<br>
Chart Or Publication; Aircraft                            1<br>
ATC Equipment / Nav Facility / Buildings; Aircraft        1<br>
Aircraft; Airspace Structure                              1<br>


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
8. Use contextual embeddings to capture deep contextual meanings

### 3. How Models work
***BERT***
![alt text](<Image Files/image.png>)
BERT changed the way machines interpret human language. Short for Bidirectional Encoder Representations from Transformers, it allows models to understand context by reading text in both directions

BERT relies on a Transformer (the attention mechanism that learns contextual relationships between words in a text). A basic Transformer consists of an encoder to read the text input and a decoder to produce a prediction for the task. Since BERT’s goal is to generate a language representation model, it only needs the encoder part.

***ROBERTA***
RoBERTa (Robustly Optimized BERT Approach) is a state-of-the-art language representation model developed by Facebook AI. It is based on the original BERT (Bidirectional Encoder Representations from Transformers) architecture but differs in several key ways.

![alt text](<Image Files/image-1.png>)

One of the ways is dynamic masking which involves randomly masking different tokens at different points during pre-training as compared to static masking use by BERT which uses the same mask every time

In the original BERT model, the pre-training phase includes a next-sentence prediction (NSP) task, where the model is trained to predict whether a given sentence is the next sentence in a text or not.In RoBERTa, this NSP loss is not used during pre-training. RoBERTa is able to learn a more reliable representation of the language by training the model on complete sentences as opposed to sentence pairs.

***AEROBERT***
A BERT model which has been pretrained on aeronautical terms

#### 4. Modeling & Evaluation
Use confusion matrix<br>
Main Evaluation metrics: Recall, Precison, F1-score<br>
Error Analysis to see where model goes wrong<br>

### Authors & Acknowledgments
Victor Ouma - Data Scientist <br>
Thanks to Zindua School for guidance during this capstone project.






