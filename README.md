# Recommender Systems Project  
**Collaborative Filtering, PCA Variants, and Domain-Specific Hybrid Recommendation**

This repository presents a comprehensive recommender systems project that combines **algorithmic experimentation** with **end-to-end system design and evaluation**.  
It covers classical and advanced recommendation techniques, including **SVD**, **PCA-based methods**, **Content-Based Filtering**, **Collaborative Filtering**, and **Hybrid Approaches**, applied to both **explicit rating datasets** and a **real-world remote job matching domain**.

---

## 📌 Project Scope

The project consists of two tightly connected components:

1. **Collaborative Filtering Experiments (SVD + PCA)**  
   Focused on understanding matrix factorization and dimensionality reduction techniques on explicit user–item rating data.

2. **Remote Job Matching Recommender System**  
   A full domain-specific recommender system integrating content-based, collaborative, and hybrid models with rigorous evaluation and cold-start analysis.

Together, these components provide both **theoretical grounding** and **practical system implementation**.

---

## 🧠 Part A: Collaborative Filtering Experiments (SVD + PCA)

This part explores collaborative filtering techniques using MovieLens-style explicit rating datasets.

### Implemented Methods

- Singular Value Decomposition (SVD) for collaborative filtering  
- PCA with Mean Filling  
- PCA with Maximum Likelihood Estimation (MLE)  

### Notebooks

- `Singular_Value_Decomposition_(SVD)_for_Collaborative_Filtering.ipynb`
- `PCA_Method_with_Mean_Filling.ipynb`
- `PCA_Method_with_Maximum_Likelihood_Estimation.ipynb`

Recommended execution order: **SVD → PCA (Mean Filling) → PCA (MLE)**

### Expected Input Data

A ratings table with the following columns:

- `userId` (int)  
- `movieId` (int)  
- `rating` (float)  

Typical filename: `ratings.csv`

### Outputs

All notebooks write results to an `out/` directory, including:

- User and item statistics  
- Popularity and grouping analysis  
- Target-user prediction outputs  
- Reconstruction error logs (observed vs. full matrix)  
- Intermediate preprocessing artifacts (pickled files)

---

## 💼 Part B: Remote Job Matching Recommender System

This part applies recommender system techniques to a real-world domain: **remote job recommendation** using the Workana dataset.

### Key Features

- **Content-Based Filtering**
  - TF-IDF vectorization
  - Cosine similarity
- **Collaborative Filtering**
  - Item-based CF with k-NN (k = 20)
  - Matrix Factorization using SVD (k = 20)
- **Hybrid Recommendation**
  - Weighted combination of content-based and collaborative scores
  - Min–max normalization
- **Evaluation**
  - Leave-One-Out protocol
  - Cold-start robustness analysis
- **Baselines**
  - Random
  - Popularity-based

---

## 📊 Dataset Overview (Remote Job Domain)

- Users: 42,085  
- Items: 21,017 job postings  
- Interactions: 96,081  
- Rating Scale:
  - Purchase → 5  
  - Click → 3  
  - Other → 1  

Implicit feedback is converted into explicit ratings to allow consistent evaluation.

---

## 📁 Project Structure

```
.
├── notebooks/
│   ├── SVD_for_CF.ipynb
│   ├── PCA_Mean_Filling.ipynb
│   └── PCA_MLE.ipynb
│
├── code/
│   ├── data_preprocessing.py
│   ├── content_based.py
│   ├── collaborative.py
│   ├── hybrid.py
│   └── main.py
│
├── data/
│   ├── ratings.csv
│   └── processed/
│       ├── items.csv
│       └── interactions.csv
│
├── results/
│   ├── metrics_table.csv
│   ├── cold_start_table.csv
│   └── *.png
│
├── out/
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Create Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Notebooks (Experiments)

```bash
jupyter notebook
```

### 4. Run Full System Pipeline (Domain Recommender)

```bash
cd code
python data_preprocessing.py
python main.py
```

---

## 🔬 Evaluation Methodology

### Leave-One-Out Evaluation

For each user:
1. Hold out one relevant interaction as test  
2. Train on remaining interactions  
3. Recommend top-10 unseen items  
4. Compute metrics  

### Metrics

- Precision@10  
- Recall@10  
- Hit@10  

### Cold-Start Evaluation

- Simulated limited user history:
  - 3 interactions
  - 5 interactions
  - 10 interactions

---

## 📈 Key Results Summary

- Item-Based CF achieved the best overall performance  
- Hybrid approach provided a strong balance between robustness and accuracy  
- Content-Based and SVD showed similar performance on sparse implicit data  
- Random and Popularity baselines performed significantly worse  

---

## 🎓 Academic Context

This repository corresponds to **Section 2 (Domain Recommender System)** of the *Intelligent Recommender Systems* course and includes:

- Data preprocessing and analysis  
- Multiple recommendation algorithms  
- Rigorous evaluation methodology  
- Cold-start analysis  

---

## 📝 License

Developed for academic and educational purposes.

---

## ✅ Project Status

Complete and ready for submission.
