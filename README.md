# Financial-News-Articles

## Introduction
This project focuses on applying a fine-tuned **BERT-based model** for sentiment analysis on financial news articles. With the increasing importance of Natural Language Processing (NLP) in understanding financial markets, this project aims to classify financial tweets into three sentiment categories: **Bearish**, **Bullish**, and **Neutral**. 

The results, while not perfect, provide valuable insights into the challenges and limitations of NLP applications in the financial domain.

### Problem and Motivation
- Financial sentiment is crucial for understanding market trends, predicting stock movements, and making informed decisions. However, the nuanced and domain-specific language of financial tweets presents significant challenges for traditional NLP models.
- This project aims to showcase the potential of fine-tuning pre-trained language models for financial sentiment classification, identify key challenges, and propose future directions for improvement.

### Key Questions
1. How effectively can a fine-tuned BERT model classify financial sentiment in tweets?
2. What are the key challenges in applying NLP techniques to financial sentiment analysis?
3. What improvements can be made to enhance the accuracy and reliability of the model?

---

## Dataset

### Description
The **Twitter Financial News Dataset** is an annotated corpus of finance-related tweets designed for multi-class sentiment classification. The dataset contains the following sentiment categories:

| Sentiment      | Description                                   |
|----------------|-----------------------------------------------|
| **Bearish (LABEL_0)** | Negative sentiment about financial markets.  |
| **Bullish (LABEL_1)** | Positive sentiment about financial markets.  |
| **Neutral (LABEL_2)** | Objective, fact-based sentiment.          |

- **Number of Instances**: 11,932 tweets.
- **Splits**:
  | Split       | Instances |
  |-------------|-----------|
  | Train       | 9,938     |
  | Validation  | 2,486     |

### Licensing Information
The dataset is released under the **MIT License**, making it suitable for research and educational purposes.

---

## Model Architecture

### Description
The project uses **BERT-base-uncased** from Hugging Face, a pre-trained transformer model fine-tuned for multi-class sentiment classification. The model takes financial tweets as input and predicts one of the three sentiment labels: **Bearish**, **Bullish**, or **Neutral**.

### Training Configuration
- **Optimizer**: AdamW
- **Learning Rate Scheduler**: Linear decay
- **Epochs**: 3
- **Batch Size**: 16
- **Loss Function**: Cross-entropy

### Performance
- **Training Loss**: 0.8455
- **Evaluation Loss**: 0.5397
- **Observations**:
  - Neutral tweets were classified most accurately.
  - The model struggled with Bearish and Bullish sentiments, likely due to class imbalance.

---

## Analysis and Results

### Learning Curve
The training loss decreased consistently across epochs, demonstrating that the model was learning effectively:

![image](https://github.com/user-attachments/assets/e5163755-7f19-467b-9432-925d4dceaf64)


### Confusion Matrix
The confusion matrix reveals the model’s performance for each sentiment class:

![image](https://github.com/user-attachments/assets/99dea4f2-0505-47ce-aad5-15701528ed52)


- **Neutral (LABEL_2)**: Most accurately classified.
- **Bearish (LABEL_0)**: Frequently misclassified as Neutral or Bullish.
- **Bullish (LABEL_1)**: Often confused with Neutral.

### Class Distribution
The imbalanced distribution of sentiment classes in the training dataset likely impacted the model’s performance:

![image](https://github.com/user-attachments/assets/8fc54093-31fb-4ef5-aedc-732ccc149bbe)


---


## Critical Analysis: Impact, Insights, and Next Steps

### What is the impact of this project?
This project demonstrates the potential of NLP and fine-tuned transformer models to analyze sentiment in financial news. By classifying tweets into Bearish, Bullish, and Neutral sentiments, it highlights how machine learning can assist in understanding public opinion and market trends. For financial analysts, investors, or institutions, this kind of tool could offer valuable insights into sentiment-driven market movements.

However, the project also underscores the challenges of applying NLP to niche domains like finance. The nuanced language in financial texts requires a model to distinguish subtle differences in tone and context, making sentiment analysis far from straightforward. This project's results reflect both the promise and the limitations of current transformer models in addressing these complexities.

### What does this project reveal or suggest?
1. **Class Imbalance Affects Performance**:
   - Neutral tweets dominate the dataset, biasing the model towards predicting Neutral more frequently. This suggests the need for better-balanced datasets or techniques like weighted loss functions during training.

2. **Nuance in Financial Language**:
   - Financial sentiment is often subtle, relying on specific domain knowledge. For example, words like "volatile" or "rally" can have different meanings depending on the context. This reveals the potential benefit of incorporating domain-specific embeddings or lexicons in future iterations.

3. **Transformer Models Show Promise**:
   - Despite limitations, the fine-tuned BERT model successfully captured the sentiment for Neutral tweets with high accuracy, showing that transformer models are well-suited for NLP tasks requiring contextual understanding.

4. **Generalization Remains a Challenge**:
   - The small dataset size and the lack of diversity in financial topics highlight the importance of larger and more varied datasets to improve the model’s generalizability.

### What is the next step?
1. **Address Class Imbalance**:
   - Implement data augmentation techniques, such as oversampling minority classes or using synthetic data generation, to balance the dataset and reduce bias.

2. **Expand the Dataset**:
   - Collect a larger and more diverse dataset using the Twitter API, focusing on various financial topics, regions, and contexts to enhance the model's robustness.

3. **Enhance the Model**:
   - Explore domain-specific pre-trained models like FinBERT or apply additional fine-tuning on a finance-specific corpus.
   - Experiment with explainability tools like SHAP or LIME to make model predictions more interpretable.

4. **Build Interactive Applications**:
   - Develop a user-friendly interface (e.g., using Gradio or Streamlit) to make the model accessible for real-world use cases, such as financial news monitoring or investor sentiment analysis.

5. **Collaborate with Domain Experts**:
   - Partnering with financial analysts or economists could help fine-tune the model further by identifying domain-specific features or metrics that enhance prediction accuracy.

---

## Files in This Repository

1. **`README.md`**: Documentation and project report.
2. **`main.ipynb`**: Jupyter Notebook containing code for data preprocessing, model fine-tuning, and evaluation.
3. **`requirements.txt`**: List of dependencies required for reproducing the project.
4. **`prediction_results.csv`**: File containing the predicted and true labels for the validation set.
5. **`prediction_results.txt`**: A human-readable file summarizing the predictions.

---

## Future Directions

### Enhancements
1. **Interactive Demo**:
   - Develop a live prediction interface using tools like Streamlit or Gradio to demonstrate the model’s capabilities.
   - Allow users to input custom tweets and view predicted sentiment in real time.
   
2. **Expanded Dataset**:
   - Collect additional finance-related tweets using the Twitter API for improved generalization.
   - Address class imbalance by augmenting minority classes through oversampling or synthetic data generation.
   
3. **Explainability and Transparency**:
   - Provide detailed explanations of model predictions using tools like SHAP or LIME to enhance trust and usability.
   - Include feature importance visualizations to identify which words or phrases most influence sentiment classification.

---

## Resource Links

1. **Dataset**: [Hugging Face Twitter Financial News Dataset](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)
2. **Model**: [BERT-base-uncased on Hugging Face](https://huggingface.co/bert-base-uncased)
3. **Research Papers**:
   - Vaswani, A., et al. (2017). *Attention is All You Need*. [Link](https://arxiv.org/abs/1706.03762)
   - Araci, D. (2019). *FinBERT: A Pretrained Language Model for Financial Communications*. [Link](https://arxiv.org/abs/1908.10063)

---

