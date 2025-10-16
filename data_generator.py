# data_generator.py

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_moons, make_circles

class SyntheticDataGenerator:
    """
    A class to generate a variety of synthetic tabular datasets for classification.
    The goal is to produce a diverse set of problems for pre-training a 
    general-purpose tabular model.
    """

    def __init__(self, 
                 min_samples=200, max_samples=1024,
                 min_features=5, max_features=50,
                 min_classes=2, max_classes=10):
        """
        Initializes the generator with constraints for the datasets.
        
        Args:
            min_samples (int): Minimum number of samples in a dataset.
            max_samples (int): Maximum number of samples. This is important for
                               managing GPU memory, as sequence length in Transformers
                               is the number of samples.
            min_features (int): Minimum number of features.
            max_features (int): Maximum number of features.
            min_classes (int): Minimum number of target classes.
            max_classes (int): Maximum number of target classes.
        """
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.min_features = min_features
        self.max_features = max_features
        self.min_classes = min_classes
        self.max_classes = max_classes

    def _generate_clustered_data(self, n_samples, n_features, n_classes):
        """
        Generates data with hyper-elliptical clusters using scikit-learn's
        make_classification. This is the workhorse function.
        """
        # Randomize the number of informative, redundant, and repeated features
        n_informative = np.random.randint(2, n_features + 1)
        n_redundant = np.random.randint(0, n_features - n_informative + 1)
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_classes=n_classes,
            n_clusters_per_class=np.random.randint(1, 4), # Creates more complex structures
            class_sep=np.random.uniform(0.5, 2.0),       # Controls how easy the problem is
            flip_y=np.random.uniform(0.01, 0.1),         # Adds label noise
            random_state=None # Ensure variety in each call
        )
        return X, y

    def _generate_nonlinear_data(self, n_samples):
        """
        Generates data with non-linear patterns (moons or circles).
        Note: These are inherently 2D, so we will add noise features later.
        """
        # 50/50 chance of generating moons or circles
        noise = np.random.uniform(0.05, 0.2)
        if np.random.rand() > 0.5:
            X, y = make_moons(n_samples=n_samples, noise=noise, random_state=None)
        else:
            X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=None)
        return X, y

    def _add_irrelevant_features(self, X, n_total_features):
        """Adds random Gaussian noise features to the dataset."""
        n_current_features = X.shape[1]
        if n_current_features >= n_total_features:
            return X
        
        n_noise_features = n_total_features - n_current_features
        noise = np.random.randn(X.shape[0], n_noise_features)
        
        return np.hstack((X, noise))

    def generate(self):
        """
        The main method to generate a single synthetic dataset.
        It randomly picks a generation strategy and then post-processes the data.
        
        Returns:
            (pd.DataFrame, pd.Series): A tuple of features (X) and target (y).
        """
        # 1. Randomly determine the properties of the new dataset
        n_samples = np.random.randint(self.min_samples, self.max_samples + 1)
        n_features = np.random.randint(self.min_features, self.max_features + 1)
        n_classes = np.random.randint(self.min_classes, self.max_classes + 1)
        
        # 2. Choose a generation method (80% chance for clustered, 20% for non-linear)
        if np.random.rand() < 0.8:
            X, y = self._generate_clustered_data(n_samples, n_features, n_classes)
        else:
            # Non-linear is always 2 classes for simplicity
            X, y = self._generate_nonlinear_data(n_samples)
            # Since non-linear starts as 2D, we must add features to match n_features
            X = self._add_irrelevant_features(X, n_features)

        # 3. Final shuffle of the data to ensure randomness
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]

        # 4. Convert to a pandas DataFrame for easier inspection (optional)
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_s = pd.Series(y, name='target')
        
        return X_df, y_s

# --- HOW TO USE IT ---
if __name__ == '__main__':
    # Create an instance of the generator
    generator = SyntheticDataGenerator()

    print("Generating a sample dataset...")
    # Generate one dataset
    X_train, y_train = generator.generate()

    # Inspect the output
    print("\nDataset properties:")
    print(f"Number of samples: {X_train.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of classes: {len(y_train.unique())}")
    
    print("\nFirst 5 rows of features (X):")
    print(X_train.head())
    
    print("\nFirst 5 labels (y):")
    print(y_train.head())
    
    print("\nClass distribution:")
    print(y_train.value_counts())