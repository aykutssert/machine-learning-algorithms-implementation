# Machine Learning Algorithms Implementation

A comprehensive collection of machine learning algorithms implemented from scratch and applied to real-world datasets, demonstrating fundamental ML concepts including supervised learning, unsupervised learning, ensemble methods, and neural networks.

## 🎯 Project Overview

This repository contains five progressive assignments that explore core machine learning algorithms and techniques. Each implementation focuses on understanding the mathematical foundations while providing practical applications on standard datasets including UCI Machine Learning Repository data, MNIST digits, and real-world classification/regression problems.

### Learning Objectives
- **Algorithm Implementation**: Building ML algorithms from mathematical foundations
- **Performance Evaluation**: Cross-validation, confusion matrices, ROC curves
- **Feature Engineering**: Dimensionality reduction, principal component analysis
- **Ensemble Methods**: Random forests, AdaBoost, neural network ensembles
- **Clustering Analysis**: K-means with multiple distance metrics

## 📚 Assignment Breakdown

### HW1: Foundational ML Algorithms - KNN, SVM, Decision Trees
**Datasets**: Breast Cancer Wisconsin (Classification), Bike Sharing (Regression)

#### Classification Tasks
- **K-Nearest Neighbors (K=3)**: Custom implementation with Euclidean distance
- **Support Vector Machine**: Linear SVM with optimal threshold selection via ROC analysis
- **Decision Trees**: Implementation with two pruning strategies and rule extraction

#### Regression Tasks  
- **KNN Regression (K=3)**: Custom implementation with Manhattan distance
- **SVM Regression**: Linear SVR for continuous target prediction
- **Decision Tree Regression**: Tree-based regression with rule extraction

**Key Features:**
- Custom distance functions (Euclidean, Manhattan)
- 6-fold cross-validation with performance metrics
- Runtime performance analysis
- Rule extraction from decision trees
- ROC curve analysis for optimal thresholds

### HW2: Advanced Decision Trees and Random Forests
**Dataset**: Abalone Classification (UCI Repository)

#### Core Implementations
- **Decision Tree Builder**: `build_dt(X, y, attribute_types, options)`
  - Handles mixed data types (numeric and categorical features)
  - Information gain-based splitting criteria
  - No pruning baseline implementation

- **Decision Tree Predictor**: `predict_dt(dt, X, options)`
  - Efficient tree traversal for predictions
  - Support for mixed feature types

#### Advanced Features
- **Pruning Implementation**: Post-pruning strategy to prevent overfitting
- **Random Decision Forest**: `build_rdf(X, y, attribute_types, N, options)`
  - Ensemble of N decision trees
  - Bootstrap aggregating (bagging)
  - Feature randomness for diversity

**Performance Evaluation:**
- K-fold cross-validation with confusion matrices
- Comparison of pruned vs unpruned trees
- Ensemble performance analysis

### HW3: Principal Component Analysis and Dimensionality Reduction
**Dataset**: MNIST Digit Recognition (60,000 training images)

#### PCA Implementation
- **Custom PCA Function**: `pca(X)` using only SVD decomposition
  - Returns mean, eigenvalues (weights), and eigenvectors
  - No external PCA libraries used
  - Mathematical foundation from singular value decomposition

#### Dimensionality Reduction Analysis
- **Component Selection**: Testing multiple dimensionality levels
- **2D Visualization**: Plotting data in first two principal components
- **Alternative Views**: First vs third principal component analysis
- **Classification Pipeline**: PCA → Random Forest with cross-validation

**Key Insights:**
- Variance explained by principal components
- Clustering visualization in reduced dimensions
- Performance trade-offs with dimensionality reduction
- Feature importance in principal component space

### HW4: Clustering with K-Means and Multiple Distance Metrics
**Dataset**: MNIST Digit Recognition (Unsupervised Learning)

#### Distance Metrics Comparison
- **L2 Norm (Euclidean)**: Standard Euclidean distance clustering
- **L1 Norm (Manhattan)**: City-block distance for robust clustering  
- **Cosine Distance**: Angle-based similarity for high-dimensional data

#### Clustering Evaluation Framework
- **Cluster Labeling Strategy**: Maximum assignment based labeling
  - Confusion matrix construction for training data
  - Optimal cluster-to-label assignment algorithm
  - Handling multiple labels per cluster

- **Performance Metrics**:
  - Training accuracy via confusion matrix
  - Test accuracy using 1-NN cluster assignment
  - 5-fold cross-validation for robust evaluation

**Advanced Analysis:**
- 80/20 train-test split with cross-validation
- Cluster purity and separation analysis
- Distance metric impact on clustering quality

### HW5: Neural Networks and Ensemble Learning
**Dataset**: UCI Repository (Student Selected)

#### Neural Network Implementation
- **Multi-Layer Perceptron**: Single hidden layer architecture
- **AdaBoost Integration**: Neural networks as base classifiers
  - Adaptive boosting with MLP weak learners
  - Weighted error minimization
  - Sequential learning with sample reweighting

#### Random Decision Forest Enhancement
- **Perceptron-Based Decisions**: Trainable perceptrons at each node
  - Replacing simple threshold decisions
  - Learning optimal decision boundaries
  - Non-linear decision surfaces in trees

**Innovation Highlights:**
- Hybrid ensemble methods (AdaBoost + Neural Networks)
- Advanced tree architectures with learned decisions
- Performance comparison with traditional methods

## 🛠️ Technical Implementation

### Core Technologies
- **Python**: Primary implementation language
- **NumPy**: Matrix operations and mathematical computations
- **Scikit-learn**: Performance benchmarking and validation
- **Matplotlib**: Data visualization and result plotting
- **Jupyter Notebooks**: Interactive development and reporting

### Mathematical Foundations
- **Linear Algebra**: Eigenvalue decomposition, SVD, matrix operations
- **Optimization**: Gradient descent, convex optimization
- **Statistics**: Cross-validation, hypothesis testing, performance metrics
- **Information Theory**: Entropy, information gain, mutual information

### Custom Implementations
- Distance functions (Euclidean, Manhattan, Cosine)
- Decision tree splitting criteria
- PCA via singular value decomposition
- K-means clustering with multiple metrics
- Neural network forward/backward propagation

## 📊 Performance Evaluation

### Validation Strategies
- **K-fold Cross-Validation**: Robust performance estimation
- **Confusion Matrices**: Detailed classification analysis
- **ROC Curves**: Threshold optimization for binary classification
- **Runtime Analysis**: Computational efficiency measurement

### Metrics and Visualizations
- **Classification**: Accuracy, Precision, Recall, F1-score
- **Regression**: MSE, MAE, R-squared
- **Clustering**: Silhouette score, cluster purity, separation
- **Dimensionality Reduction**: Variance explained, reconstruction error

## 🎓 Learning Outcomes

This project demonstrates mastery of:
- **Algorithm Development**: Implementing ML algorithms from mathematical principles
- **Data Analysis**: Working with real-world datasets and handling various data types
- **Performance Optimization**: Cross-validation, hyperparameter tuning, ensemble methods
- **Mathematical Modeling**: Understanding statistical and algebraic foundations
- **Software Engineering**: Clean code, modular design, reproducible experiments
- **Research Skills**: Experimental design, hypothesis testing, result interpretation

## 📁 Project Structure

```
machine-learning-algorithms-implementation/
├── HW1_KNN_SVM_DecisionTree.ipynb           # Foundational algorithms
├── HW2_DecisionTree_RandomForest.ipynb      # Advanced tree methods
├── HW3_PCA_DimensionalityReduction.ipynb    # Feature reduction techniques
├── HW4_Clustering_KMeans.ipynb              # Unsupervised learning
├── HW5_NeuralNetworks_AdaBoost.ipynb        # Neural networks & ensembles
└── README.md                                # This documentation
```

## 🔬 Research Applications

### Industry Relevance
- **Computer Vision**: MNIST digit recognition techniques
- **Healthcare Analytics**: Medical diagnosis via classification algorithms
- **Financial Modeling**: Regression analysis for market prediction
- **Recommendation Systems**: Clustering for user segmentation
- **Automated Decision Making**: Rule extraction from decision trees

### Academic Contributions
- Comparative analysis of distance metrics in clustering
- Performance evaluation of hybrid ensemble methods
- Dimensionality reduction impact on classification accuracy
- Custom implementation insights vs. library performance

## 🎯 Technical Highlights

- **Mathematical Rigor**: Algorithms implemented from first principles
- **Comprehensive Evaluation**: Multiple datasets, metrics, and validation strategies
- **Scalable Implementation**: Efficient algorithms suitable for large datasets
- **Reproducible Research**: Clear documentation and experimental protocols
- **Professional Code Quality**: Well-structured, commented, and modular implementations

---

**Course**: CSE455/CSE552 Machine Learning  
**Term**: Spring 2025  
**Skills Demonstrated**: Algorithm Implementation, Statistical Analysis, Data Science, Mathematical Modeling, Software Development