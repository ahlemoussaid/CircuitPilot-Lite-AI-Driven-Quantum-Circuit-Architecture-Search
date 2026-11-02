import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_dataset(n_features: int = 4, seed: int = 42) -> dict:
    """Breast cancer data reduced to n_features, split 60/20/20 into train/val/test.

    Features are scaled to [0, pi] so each one can be used directly as a rotation
    angle. Labels are mapped to -1 (malignant) and +1 (benign) to match the range
    of a Pauli-Z expectation value.
    """
    X, y = load_breast_cancer(return_X_y=True)
    y = 2 * y - 1

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=seed
    )

    # All preprocessing is fitted on the training split only, so no information
    # from validation or test data leaks into the features.
    standardizer = StandardScaler().fit(X_train)
    pca = PCA(n_components=n_features, random_state=seed).fit(standardizer.transform(X_train))
    angle_scaler = MinMaxScaler(feature_range=(0, np.pi)).fit(
        pca.transform(standardizer.transform(X_train))
    )

    def transform(X_raw):
        X_reduced = angle_scaler.transform(pca.transform(standardizer.transform(X_raw)))
        # Val/test points can fall slightly outside the training range.
        return np.clip(X_reduced, 0, np.pi)

    return {
        "X_train": transform(X_train),
        "y_train": y_train,
        "X_val": transform(X_val),
        "y_val": y_val,
        "X_test": transform(X_test),
        "y_test": y_test,
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
    }
