# Robot-Navigation-Based-on-Machine-Learning

## Project Overview

This project focuses on building a machine learning-based system to predict robot movement directions using real-time sensor input data. Instead of using rule-based navigation, which is often rigid and fails in dynamic environments, we propose a model that can generalize and make intelligent decisions by learning from data. The goal is to help autonomous robots navigate efficiently and safely using predictive modeling. The final model is optimized for real-world deployment by improving its performance through preprocessing and hyperparameter tuning.

## Objective

Our objective was to explore various machine learning classifiers and identify the one that performs best for predicting robot movements. The task involves using sensor data from the Wall Following Robot Navigation dataset and determining the robot’s movement direction—forward, left, right, or sharp right—based on the sensor readings. We compared different models, tuned the hyperparameters, and evaluated them with performance metrics, especially the F1-score, due to class imbalance in the dataset.

## Why This Project?

Autonomous navigation is at the heart of modern robotics applications, ranging from warehouse automation to assistive robotics and self-driving systems. Traditional methods rely on if-else logic or hard-coded instructions, which are not adaptable to unseen situations or sensor noise. By leveraging machine learning, we can train a robot to make informed decisions based on patterns in sensor data, making it more robust in dynamic environments. The dataset used here replicates real-world sensor interactions, making this a practical and meaningful ML application.

## Dataset Overview

We used the Wall Following Robot Navigation dataset, which includes 4,910 training samples and 546 test samples. Each sample contains 24 features (V1 through V24) representing ultrasonic sensor readings. The target variable is a movement class with one of four possible values (1, 2, 3, 4), corresponding to specific directions.

Before training, we checked for missing values, performed imputation as necessary, and applied feature scaling to normalize the data. This helped improve model performance, especially for distance-based algorithms like KNN.

## Class Distribution

We observed that the dataset is imbalanced across the four movement classes. Class 1 (Move Forward) and Class 2 (Move Left) had the highest frequency, while Class 3 (Move Right) had the fewest samples. This imbalance made it crucial to choose evaluation metrics carefully. Accuracy alone could be misleading, so we relied on the F1-score to ensure balanced performance across all classes.

## Exploratory Data Analysis

To better understand the data, we conducted exploratory analysis that included class distribution visualization, feature-wise histograms, and correlation analysis. None of the features were highly correlated (correlation threshold set at 0.9), so we retained all features for training. This analysis confirmed data quality and guided our modeling strategy.

## Models Evaluated

We selected four classifiers that are commonly used for multi-class classification problems:

- **Support Vector Machine (SVM)**: Chosen for its robustness in high-dimensional spaces and strong generalization capabilities.
- **Random Forest**: Selected for its ability to handle noisy data and non-linear decision boundaries.
- **Gradient Boosting**: A powerful ensemble method known for accuracy and minimizing overfitting.
- **K-Nearest Neighbors (KNN)**: Simple and interpretable, useful as a baseline model.

These models strike a balance between interpretability, accuracy, and computational efficiency, making them ideal candidates for the robot navigation task.

## Why Not Neural Networks or Regression?

We deliberately chose not to use neural networks due to their high computational cost, need for large datasets, and tendency to overfit on smaller datasets like ours. Similarly, regression algorithms were excluded because our problem is classification-based. Regression models such as Linear Regression predict continuous values, whereas our goal was to predict discrete classes representing direction.

## Baseline Model Performance

We initially trained all models using default hyperparameters to establish baseline performance. Among them, Random Forest and Gradient Boosting performed best even before tuning. KNN struggled due to its reliance on distance metrics, which can be sensitive to scale and noise. These initial results helped us understand each model’s strengths and limitations.

## Hyperparameter Tuning

To improve performance, we applied hyperparameter tuning using `GridSearchCV` with 5-fold cross-validation. This allowed us to fine-tune parameters such as kernel type and regularization (SVM), depth and number of trees (Random Forest), learning rate (Gradient Boosting), and number of neighbors (KNN). All models showed performance gains after tuning, especially in terms of F1-score.

## Best Parameters and Results

After tuning, the Random Forest classifier achieved the highest F1-score of **99.63%**, making it the top-performing model. SVM and Gradient Boosting also performed well, with scores around 94% and 99.4%, respectively. KNN improved but still lagged behind. The Random Forest model was ultimately selected for deployment due to its accuracy, stability, and interpretability.

## Evaluation Metric: Why F1-Score?

We chose F1-score as the primary evaluation metric because the dataset was imbalanced. While accuracy may show high performance due to the dominant classes, it doesn’t reflect how well the model performs across all classes. F1-score, which balances precision and recall, provided a more reliable measure of performance, especially for minority classes.

## Final Model: Random Forest

The final model we selected is a tuned Random Forest classifier. It generalizes well to unseen test data and has minimal misclassifications. This made it the most robust option for real-world application. We further confirmed this by analyzing the confusion matrix, which showed perfect classification for major classes and very low errors for the smallest class.

## Learning Curve Analysis

We plotted the learning curve for the Random Forest model to verify that it wasn’t overfitting. The training F1-score remained high at 100%, while the validation F1-score steadily converged to 99.63%. This confirmed that the model had strong generalization capability and neither underfit nor overfit the data.

## Feature Importance

We examined feature importance to determine which sensor inputs had the most impact on model predictions. Features like V15, V19, and V20 emerged as the most significant. Understanding feature contributions can help in optimizing future robotic sensor configurations and reducing hardware complexity.

## Performance Results and Best Model

We saved the performance results in a json and best model in a pkl file to use it in the future.

## Conclusion

This project successfully demonstrated that machine learning can be effectively applied to robot navigation using real-world sensor data. After evaluating and tuning multiple classifiers, we found that the Random Forest model offers the best performance with an F1-score of 99.63%. Through careful data preprocessing, hyperparameter tuning, and metric selection, we were able to build a robust and highly accurate model for predicting robot movement.

##  Required Packages

The following packages are required:
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib

```bash
# Install necessary Python packages
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

##  Running

To run the project you just need to run the cells of the jupyter notebook file. Please make sure the dataset is in the same folder as the ipynb file.
