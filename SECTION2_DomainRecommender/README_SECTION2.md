# Remote Job Matching Recommender System

A comprehensive recommender system for remote job matching, implementing content-based filtering, collaborative filtering (Item-CF and SVD), and hybrid approaches with extensive evaluation.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [Files Description](#files-description)

---

## 🎯 Project Overview

This project implements a domain-specific recommender system for remote job matching using the Workana dataset. The system converts implicit user feedback (clicks, purchases) into explicit ratings and provides personalized job recommendations using multiple approaches.

### Key Features

- **Content-Based Filtering**: TF-IDF vectorization with cosine similarity
- **Collaborative Filtering**: 
  - Item-based CF with k-NN (k=20)
  - Matrix Factorization using SVD (k=20)
- **Hybrid Approach**: Weighted combination with min-max normalization
- **Comprehensive Evaluation**: Leave-One-Out methodology with cold-start analysis
- **Baseline Comparisons**: Random and Popularity baselines

### Dataset Statistics

- **Users**: 42,085
- **Items**: 21,017 job postings
- **Interactions**: 96,081
- **Rating Scale**: 1-5 (purchase=5, click=3, other=1)

---

## 📁 Project Structure

```
SECTION2_DomainRecommender/
├── code/
│   ├── data_preprocessing.py    # Data loading and preprocessing
│   ├── content_based.py         # Content-based filtering implementation
│   ├── collaborative.py         # Item-CF and SVD implementation
│   ├── hybrid.py                # Hybrid recommendation system
│   └── main.py                  # Main evaluation pipeline
├── data/
│   └── processed/
│       ├── items.csv            # Processed job items (37.6 MB)
│       └── interactions.csv     # User-item interactions (2.7 MB)
├── results/
│   ├── metrics_table.csv        # Overall performance metrics
│   ├── cold_start_table.csv     # Cold-start evaluation results
│   ├── metrics_comparison.png   # Performance comparison chart
│   ├── rating_distribution.png  # Rating distribution visualization
│   ├── user_activity_distribution.png
│   └── item_popularity_distribution.png
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🔧 Requirements

### Python Version
- Python 3.8 or higher

### Dependencies

```
numpy
pandas
scikit-learn
matplotlib
scipy
```

Install all dependencies using:
```bash
pip install -r requirements.txt
```

---

## 🚀 Installation

1. **Clone or download the project**
   ```bash
   cd SECTION2_DomainRecommender
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure data files are in place**
   - Raw data should be in `../archive/` directory:
     - `event_data.csv` (clickstream data)
     - `job_information.csv` (job metadata)

---

## 💻 Usage

### Step 1: Data Preprocessing

Run the preprocessing script to clean data and generate visualizations:

```bash
cd code
python data_preprocessing.py
```

**Output:**
- `data/processed/items.csv` - Cleaned job items
- `data/processed/interactions.csv` - User-item interactions
- `results/rating_distribution.png`
- `results/user_activity_distribution.png`
- `results/item_popularity_distribution.png`

### Step 2: Run Complete Evaluation Pipeline

Execute the main evaluation script:

```bash
python main.py
```

**This will:**
1. Train Content-Based model (TF-IDF)
2. Train Item-CF model (k-NN, k=20)
3. Train SVD model (Matrix Factorization, k=20)
4. Evaluate all 6 methods using Leave-One-Out on 300 users
5. Perform cold-start evaluation (N=3, 5, 10 interactions)
6. Generate metrics tables and visualizations

**Output:**
- `results/metrics_table.csv` - Overall performance metrics
- `results/cold_start_table.csv` - Cold-start evaluation results
- `results/metrics_comparison.png` - Performance comparison chart

### Step 3: Test Individual Modules (Optional)

Test each recommender independently:

```bash
# Test Content-Based
python content_based.py

# Test Collaborative Filtering (Item-CF + SVD)
python collaborative.py

# Test Hybrid Approach
python hybrid.py
```

---

## 🔬 Implementation Details

### 1. Data Preprocessing (`data_preprocessing.py`)

- Converts implicit feedback to explicit ratings:
  - **Purchase** → Rating 5
  - **Click** → Rating 3
  - **Other** → Rating 1
- Cleans job descriptions (HTML removal)
- Combines title and description for content representation
- Aggregates multiple interactions per user-item pair (takes max rating)

### 2. Content-Based Filtering (`content_based.py`)

- **TF-IDF Vectorization**:
  - Max features: 20,000
  - N-grams: unigrams + bigrams
  - Min document frequency: 2
- **User Profile**: Weighted average of item vectors (using ratings as weights)
- **Scoring**: Cosine similarity between user profile and item vectors
- **Item-kNN Variant**: k=10, 20 for prediction

### 3. Collaborative Filtering (`collaborative.py`)

#### Item-Based CF
- Sparse matrix implementation (CSR format)
- k-NN with cosine similarity (k=20)
- Score aggregation: Σ(similarity × user_rating)

#### SVD (Matrix Factorization)
- TruncatedSVD with k=20 components
- Works directly on sparse matrices
- Prediction: dot(user_factors, item_factors)

### 4. Hybrid Approach (`hybrid.py`)

- **Formula**: `HybridScore = α × CB_norm + (1-α) × CF_norm`
- **Normalization**: Min-max scaling to [0, 1]
- **Alpha values tested**: 0.3, 0.5, 0.7
- **Pool size**: 200 candidates from each method before combining

### 5. Evaluation (`main.py`)

#### Methodology: Leave-One-Out
- For each user with ≥1 purchase:
  1. Hold out 1 purchase (rating ≥ 5) as test
  2. Use remaining interactions for training
  3. Recommend top-10 unseen items
  4. Calculate metrics

#### Metrics
- **Precision@10**: Proportion of relevant items in top-10
- **Recall@10**: Proportion of test items found in top-10
- **Hit@10**: Binary - did we find the test item in top-10?

#### Cold-Start Evaluation
- Simulates limited user history
- Limits: 3, 5, 10 interactions
- Tests robustness with sparse data

---

## 📊 Results

### Overall Performance (Leave-One-Out, 300 users)

| Rank | Method | Precision@10 | Recall@10 | Hit@10 |
|------|--------|--------------|-----------|--------|
| 🥇 1st | **Item-CF** | 0.0587 | 0.5867 | **58.67%** |
| 🥈 2nd | **Hybrid** | 0.0477 | 0.4767 | **47.67%** |
| 🥉 3rd | Content-Based | 0.0077 | 0.0767 | 7.67% |
| 4th | SVD | 0.0073 | 0.0733 | 7.33% |
| 5th | Popularity | 0.0023 | 0.0233 | 2.33% |
| 6th | Random | 0.0000 | 0.0000 | 0.00% |

### Cold-Start Performance (Hit@10)

| Method | 3 interactions | 5 interactions | 10 interactions |
|--------|----------------|----------------|-----------------|
| Item-CF | 56.0% | 57.3% | 58.7% |
| Hybrid | 43.0% | 45.7% | 47.0% |
| Content-Based | 7.67% | 7.33% | 7.33% |
| SVD | 7.33% | 7.33% | 7.33% |
| Popularity | 2.33% | 2.33% | 2.33% |
| Random | 0.0% | 0.0% | 0.0% |

### Key Insights

✅ **Item-CF is the best performer** (58.67% Hit@10)
- Neighborhood-based collaborative filtering excels on implicit feedback
- Robust even with limited user history (56% with only 3 interactions)

✅ **Hybrid provides good balance** (47.67% Hit@10)
- Combines strengths of content and collaborative approaches
- More stable than individual methods

✅ **Content-Based and SVD have similar performance** (~7%)
- Job descriptions may have limited discriminative power
- SVD's linear factorization struggles with sparse implicit feedback

✅ **Baselines are weak** (Popularity: 2.33%, Random: 0%)
- Validates that ML-based recommenders add significant value

---

## 📄 Files Description

### Code Files

- **`data_preprocessing.py`**: Data loading, cleaning, and visualization generation
- **`content_based.py`**: TF-IDF-based content filtering with user profile construction
- **`collaborative.py`**: Item-based CF and SVD matrix factorization
- **`hybrid.py`**: Weighted combination of content-based and collaborative methods
- **`main.py`**: Complete evaluation pipeline with Leave-One-Out and cold-start analysis

### Data Files

- **`data/processed/items.csv`**: Cleaned job items with combined text (title + description)
- **`data/processed/interactions.csv`**: User-item interactions with explicit ratings

### Results Files

- **`results/metrics_table.csv`**: Overall performance metrics for all 6 methods
- **`results/cold_start_table.csv`**: Cold-start evaluation results (3 scenarios × 6 methods)
- **`results/metrics_comparison.png`**: Bar chart comparing Precision, Recall, and Hit@10
- **`results/rating_distribution.png`**: Distribution of ratings (3 and 5)
- **`results/user_activity_distribution.png`**: Histogram of interactions per user
- **`results/item_popularity_distribution.png`**: Histogram of interactions per item

---

## 🎓 Academic Context

This project implements Section 2 (Domain Recommender System) of the Intelligent Recommender Systems course project, including:

- ✅ Part 1: Data Preprocessing
- ✅ Part 2: Recommendation Algorithms (CB, CF, Hybrid)
- ✅ Part 3: Evaluation (with bonus cold-start analysis)

### Bonus Features Implemented

- ✅ Cold-start evaluation with multiple scenarios
- ✅ SVD matrix factorization in addition to Item-CF
- ✅ Comprehensive baseline comparisons
- ✅ Leave-One-Out evaluation methodology

---

## 👨‍💻 Technical Notes

### Important Implementation Details

1. **User ID Handling**: User IDs are kept as strings to avoid float corruption
2. **Sparse Matrices**: All collaborative filtering uses CSR sparse matrices (no dense 42k×21k matrices)
3. **Evaluation Protocol**: Proper Leave-One-Out with held-out test items
4. **Normalization**: Min-max scaling applied before hybrid combination

### Performance Considerations

- Training time: ~2-3 minutes for all models
- Evaluation time: ~5-7 minutes for 300 users
- Memory usage: Efficient sparse matrix operations
- Scalability: Can handle 42k users and 21k items

---

## 📝 License

This project is developed for academic purposes as part of the Intelligent Recommender Systems course.

---

## 🙏 Acknowledgments

- Dataset: Workana remote job matching platform
- Libraries: scikit-learn, pandas, numpy, scipy, matplotlib

---

**Project Status**: ✅ Complete and Ready for Submission

For questions or issues, please refer to the code documentation or contact the project maintainer.
