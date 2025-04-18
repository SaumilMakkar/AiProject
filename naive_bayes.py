import numpy as np
from collections import Counter
from config import SMOOTHING_ALPHA

class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_weights = None
        self.feature_weights = None
        self.class_likelihoods = None
        self.classes = None
        self.class_to_index = None
        
    def fit(self, X, y):
        """
        Fit Naive Bayes classifier
        
        Args:
            X (list): List of feature vectors
            y (list): List of class labels
        """
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Get unique classes and create mapping
        self.classes = np.unique(y)
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.classes)}
        n_classes = len(self.classes)
        n_features = X.shape[1]
        
        # Debug: Print dataset statistics
        print("\nDataset Statistics:")
        print(f"Number of samples: {len(X)}")
        print(f"Number of features: {n_features}")
        print(f"Number of classes: {n_classes}")
        print(f"Class distribution: {dict(Counter(y))}")
        
        # Initialize feature weights
        self.feature_weights = np.ones(n_features)
        
        # Calculate class weights (prior probabilities)
        class_counts = Counter(y)
        total_samples = len(y)
        self.class_weights = np.zeros(n_classes)
        
        for cls, idx in self.class_to_index.items():
            self.class_weights[idx] = (class_counts[cls] + self.alpha) / (total_samples + n_classes * self.alpha)
            
        # Debug: Print class weights
        print("\nClass Weights (Prior Probabilities):")
        for cls, idx in self.class_to_index.items():
            print(f"Class {cls}: {self.class_weights[idx]:.4f}")
        
        # Calculate feature likelihoods for each class
        self.class_likelihoods = np.zeros((n_classes, n_features))
        
        for cls, idx in self.class_to_index.items():
            # Get samples for current class
            class_samples = X[y == cls]
            
            # Calculate feature counts
            feature_counts = np.sum(class_samples, axis=0)
            total_words = np.sum(feature_counts)
            
            # Calculate likelihoods with smoothing
            self.class_likelihoods[idx] = (feature_counts + self.alpha) / (total_words + n_features * self.alpha)
            
            # Debug: Print class likelihood statistics
            print(f"\nClass {cls} Likelihood Statistics:")
            print(f"Number of samples: {len(class_samples)}")
            print(f"Total words: {total_words}")
            print(f"Non-zero features: {np.sum(feature_counts > 0)}")
            print(f"Average feature value: {np.mean(feature_counts):.2f}")
            
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X (list): List of feature vectors
            
        Returns:
            numpy.ndarray: Class probabilities
        """
        if self.class_weights is None or self.class_likelihoods is None:
            raise ValueError("Model not fitted. Call fit first.")
            
        if self.feature_weights is None:
            raise ValueError("Feature weights not initialized.")
            
        X = np.array(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        
        # Initialize probability matrix
        probabilities = np.zeros((n_samples, n_classes))
        
        # Calculate log probabilities for each class
        for cls, idx in self.class_to_index.items():
            # Calculate log likelihood
            log_likelihood = np.sum(X * np.log(self.class_likelihoods[idx]) * self.feature_weights, axis=1)
            
            # Add log prior
            probabilities[:, idx] = log_likelihood + np.log(self.class_weights[idx])
            
        # Debug: Print prediction statistics
        print("\nPrediction Statistics:")
        print(f"Number of samples: {n_samples}")
        print(f"Average probability range: {np.min(probabilities):.2f} to {np.max(probabilities):.2f}")
        print(f"Average probability per class: {np.mean(probabilities, axis=0)}")
        
        # Convert to probabilities using softmax
        probabilities = np.exp(probabilities - np.max(probabilities, axis=1, keepdims=True))
        probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
        
        return probabilities
        
    def predict(self, X):
        """
        Predict class labels
        
        Args:
            X (list): List of feature vectors
            
        Returns:
            list: Predicted class labels
        """
        probabilities = self.predict_proba(X)
        predictions = self.classes[np.argmax(probabilities, axis=1)]
        
        # Debug: Print prediction results
        print("\nPrediction Results:")
        print(f"Predicted class distribution: {dict(Counter(predictions))}")
        
        return predictions 