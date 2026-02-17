# AG News Text Classification - NLP Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF)](https://www.kaggle.com/code/bishaluts/ag-news-text-classification-nlp)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Project Overview

This project implements a **text classification model** to categorize news articles into 4 classes using the **AG News dataset**. The project explores traditional machine learning approaches with TF-IDF vectorization and compares multiple classification algorithms.

### Dataset Information
- **Dataset**: AG News Classification Dataset
- **Source**: [Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
- **Classes**: 4 categories
  - 🌍 **World** - International news
  - ⚽ **Sports** - Sports news
  - 💼 **Business** - Business and financial news
  - 🔬 **Sci/Tech** - Science and technology news
- **Training Samples**: 120,000
- **Test Samples**: 7,600

## 🎯 Project Goals

1. Build an effective text classification pipeline
2. Compare multiple machine learning algorithms
3. Achieve high accuracy on multi-class classification
4. Implement proper text preprocessing techniques
5. Visualize model performance with confusion matrices

## 🛠️ Technologies & Libraries

- **Python 3.8+**
- **Data Processing**: NumPy, Pandas
- **Machine Learning**: Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Text Processing**: Regular Expressions (re)

## 📊 Project Structure

```
AG-News-Text-Classification/
│
├── README.md
├── ag-news-text-classification-nlp.ipynb
└── requirements.txt (to be added)
```

## 🚀 Implementation Steps

### Step 1: Import Required Libraries
Imported essential libraries for data manipulation, visualization, and machine learning.

### Step 2: Load and Explore Dataset
- Loaded train and test CSV files
- Added proper column names
- Analyzed class distribution
- Mapped numeric classes to category names

### Step 3: Text Preprocessing
- Combined title and description fields
- Implemented text cleaning:
  - Converted to lowercase
  - Removed special characters and digits
  - Removed extra whitespace
- Prepared features (X) and labels (y)

### Step 4: Feature Engineering & Model Training
- **TF-IDF Vectorization**
  - max_features: 10,000
  - ngram_range: (1, 2) - unigrams and bigrams
  
- **Models Trained**:
  1. **Logistic Regression**
     - max_iter: 1000
     - random_state: 42
  2. **Multinomial Naive Bayes**
     - Default parameters

### Step 5: Model Evaluation
- Classification report (precision, recall, F1-score)
- Confusion matrix visualization
- Sample predictions analysis

## 📈 Results

### Model Performance
*(Results will be updated after full execution)*

| Model | Accuracy | Notes |
|-------|----------|-------|
| Logistic Regression | TBD | Expected: ~90%+ |
| Multinomial Naive Bayes | TBD | Expected: ~88%+ |

### Key Insights
- TF-IDF with bigrams captures important phrase patterns
- News articles have distinct vocabulary patterns per category
- Logistic Regression typically performs well on text classification
- The dataset is well-balanced across all 4 classes

## 💻 How to Run

### Option 1: Run on Kaggle (Recommended)
1. Visit the [Kaggle Notebook](https://www.kaggle.com/code/bishaluts/ag-news-text-classification-nlp)
2. Click "Copy & Edit" to create your own version
3. Run all cells

### Option 2: Run Locally
```bash
# Clone the repository
git clone https://github.com/bishal-uts/AG-News-Text-Classification.git
cd AG-News-Text-Classification

# Install dependencies
pip install -r requirements.txt

# Download the dataset from Kaggle
# Place train.csv and test.csv in the data/ folder

# Run Jupyter Notebook
jupyter notebook ag-news-text-classification-nlp.ipynb
```

## 🔮 Future Improvements

- [ ] Implement deep learning models (LSTM, GRU)
- [ ] Try transformer-based models (BERT, DistilBERT)
- [ ] Add word embeddings (Word2Vec, GloVe)
- [ ] Implement cross-validation
- [ ] Add hyperparameter tuning
- [ ] Create a web interface for predictions
- [ ] Deploy model as API

## 📚 Learning Outcomes

- Text preprocessing techniques for NLP
- TF-IDF vectorization and feature engineering
- Multi-class text classification
- Model comparison and evaluation
- Confusion matrix interpretation
- Scikit-learn pipeline implementation

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Bishal Sarker**
- GitHub: [@bishal-uts](https://github.com/bishal-uts)
- University: University of Technology Sydney (UTS)
- Kaggle: [bishaluts](https://www.kaggle.com/bishaluts)

## 🙏 Acknowledgments

- AG News dataset creators and Kaggle community
- Scikit-learn documentation and examples
- NLP and machine learning research papers

---

⭐ **If you find this project helpful, please consider giving it a star!**
