
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from shap_selection import feature_selection

from sklearn.metrics import accuracy_score  
from alibi.explainers import CEM
import time
import alibi

from dalex import Explainer
from sklearn.inspection import permutation_importance

from sklearn.metrics import pairwise_distances

# from ceml.contrib.explainers import CEM

from sklearn.preprocessing import StandardScaler

# This File Contains the Following Classes:
# 1. XAI_Methods
# 2. LOCO_XAI_Methods
# 3. CEM_XAI_Methods
# 4. PFI_XAI_Methods
# 5. dalex_XAI_Methods
# 6. profweight_XAI_Methods



# Shap XAI Methods 
class XAI_Methods:
    # My ref https://shap.readthedocs.io/en/latest/
    def __init__(self, model, X_train, feature_names=None):
        """
        Initialize the XAI_Methods class.

        :param model: Trained machine learning model.
        :param X_train: Training data used to compute feature importance.
        :param feature_names: Optional list of feature names. If None, will use X_train columns.
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names if feature_names else X_train.columns

    def shap_importance(self):
        """
        Compute SHAP feature importance.

        :return: DataFrame of features and their mean absolute SHAP values.
        """
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_train)

        # Aggregate SHAP values by taking mean of absolute values across samples and classes
        # Shape of shap_values: (2016638, 68, 15)
        # Shape of X_train: (2016638, 68) 
        mean_shap_values = np.mean(np.abs(shap_values), axis=(0, 2))

        feature_importance = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance": mean_shap_values
        }).sort_values(by="Importance", ascending=False)

        return feature_importance

    def plot_top_features(self, feature_importance, top_n=25):
        """
        Plot the top N features by importance.

        :param feature_importance: DataFrame of features and their importance values.
        :param top_n: Number of top features to display.
        """
        
        top_features = feature_importance.head(top_n)
        plt.figure(figsize=(12, 8))
        sns.barplot(x=top_features["Importance"], y=top_features["Feature"], palette="viridis")
        plt.title(f"Top {top_n} Features by Mean Absolute SHAP Values", fontsize=16)
        plt.xlabel("Mean Absolute SHAP Value", fontsize=14)
        plt.ylabel("Features", fontsize=14)
        plt.tight_layout()
        plt.savefig("top_features_barplot.png")  
        plt.close() 
        
        shap_values = shap.TreeExplainer(self.model).shap_values(self.X_train)

        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap.summary_plot(
            shap_values,
            features=self.X_train,
            feature_names=self.feature_names,
            max_display=top_n,
            show=False  
        )
        plt.savefig("shap_summary_plot.png")  
        plt.close()  


    def save_importance(self, feature_importance, file_path, file_format="csv"):
        """
        Save feature importance to a file.

        :param feature_importance: DataFrame of features and their importance values.
        :param file_path: Path to save the file.
        :param file_format: Format to save the file ('csv' or 'json').
        """
        if file_format == "csv":
            feature_importance.to_csv(file_path, index=False)
        elif file_format == "json":
            feature_importance.to_json(file_path, orient="records", lines=True)
        else:
            raise ValueError("Unsupported file format. Use 'csv' or 'json'.")
        print(f"Feature importance saved to {file_path} in {file_format} format.")

    # def feature_selection(self, X_test, feature_names):
    #     """
    #     Perform SHAP-based feature selection.

    #     :param X_test: Test dataset (pandas DataFrame).
    #     :param feature_names: List of feature names.
    #     :return: Ordered list of features selected by SHAP.
    #     """
    #     # Ensure X_train and X_test are NumPy arrays for compatibility with shap_select
    #     X_train_np = self.X_train.to_numpy() if not isinstance(self.X_train, np.ndarray) else self.X_train
    #     X_test_np = X_test.to_numpy() if not isinstance(X_test, np.ndarray) else X_test

    #     # Perform SHAP-based feature selection
    #     feature_order = feature_selection.shap_select(
    #         self.model, X_train_np, X_test_np, feature_names, agnostic=False
    #     )
    #     return feature_order


# --------------------------------------------------------------- LOCO (Leave-One-Covariate-Out) ---------------------------------------------------------------

# ref


class LOCO_XAI_Methods:
    def __init__(self, model, X_train, y_train, metric=accuracy_score, feature_names=None):
        """
        Initialize the XAI_Methods class for LOCO analysis.

        :param model: Trained machine learning model.
        :param X_train: Training data (features).
        :param y_train: Training labels.
        :param metric: Performance metric function (e.g., accuracy_score).
        :param feature_names: Optional list of feature names. If None, will use X_train columns.
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.metric = metric
        self.feature_names = feature_names if feature_names else X_train.columns

    def loco_importance(self):
        """
        Perform Leave-One-Covariate-Out (LOCO) analysis.

        :return: DataFrame of features and their LOCO importance.
        """
        baseline_score = self.metric(self.y_train, self.model.predict(self.X_train))  # Baseline model performance
        loco_importance = []

        for feature in self.feature_names:
            # Remove one feature
            X_train_loco = self.X_train.drop(columns=[feature])
            
            # Retrain the model on the reduced dataset
            self.model.fit(X_train_loco, self.y_train)
            reduced_score = self.metric(self.y_train, self.model.predict(X_train_loco))

            # Calculate the drop in performance
            importance = baseline_score - reduced_score
            loco_importance.append(importance)

        loco_df = pd.DataFrame({
            "Feature": self.feature_names,
            "LOCO Importance": loco_importance
        }).sort_values(by="LOCO Importance", ascending=False)

        return loco_df

    def save_importance(self, importance_df, file_path, file_format="csv"):
        """
        Save LOCO importance to a file.

        :param importance_df: DataFrame of features and their importance.
        :param file_path: Path to save the file.
        :param file_format: Format to save the file ('csv' or 'json').
        """
        if file_format == "csv":
            importance_df.to_csv(file_path, index=False)
        elif file_format == "json":
            importance_df.to_json(file_path, orient="records", lines=True)
        else:
            raise ValueError("Unsupported file format. Use 'csv' or 'json'.")
        print(f"LOCO importance saved to {file_path} in {file_format} format.")

    def plot_top_features(self, importance_df, top_n):
        """
        Plot the top features by LOCO importance.

        :param importance_df: DataFrame of features and their importance.
        :param top_n: Number of top features to display.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        
        top_features = importance_df.head(top_n)

        plt.figure(figsize=(12, 8))
        sns.barplot(x=top_features["LOCO Importance"], y=top_features["Feature"], palette="viridis")
        plt.title(f"Top {top_n} Features by LOCO Importance", fontsize=16)
        plt.xlabel("LOCO Importance", fontsize=14)
        plt.ylabel("Features", fontsize=14)
        plt.tight_layout()
        plt.show()



# -------------------------------------------------------------- CEM (Counterfactual Explanation Method) --------------------------------------------------------------

# ref 



class CEM_XAI_Methods:
    def __init__(self, model, X_train, feature_names=None):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names if feature_names else X_train.columns

    def cem_importance(self, instance_idx, mode="PN", kappa=0.1, beta=0.1, gamma=1.0):
        # 1. Handle Missing Values (Example: Simple Imputation)
        X_train_imputed = self.X_train.fillna(self.X_train.mean()) 

        # 2. Feature Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)

        input_shape = (X_train_scaled.shape[1],) 
        explainer = CEM(self.model, mode=mode, shape=input_shape) 
        explainer.fit(
            X_train_scaled, 
            no_info_val=0.0, 
            feature_range=(X_train_scaled.min(axis=0), X_train_scaled.max(axis=0))
        )

        instance = X_train_scaled[instance_idx].reshape(1, -1)
        try:
            explanation = explainer.explain(instance, kappa=kappa, beta=beta, gamma=gamma)
            feature_importance = explanation['feature_importance']
            importance_df = pd.DataFrame({
                "Feature": self.feature_names,
                "Importance": feature_importance
            })
            return importance_df
        except Exception as e:
            print(f"Error during explanation: {e}")
            return None

    def save_importance(self, feature_importance, file_path, file_format="csv"):
        """
        Save feature importance to a file.

        :param feature_importance: DataFrame of features and their importance values.
        :param file_path: Path to save the file.
        :param file_format: Format to save the file ('csv' or 'json').
        """
        if file_format == "csv":
            feature_importance.to_csv(file_path, index=False)
        elif file_format == "json":
            feature_importance.to_json(file_path, orient="records", lines=True)
        else:
            raise ValueError("Unsupported file format. Use 'csv' or 'json'.")
        print(f"Feature importance saved to {file_path} in {file_format} format.")



# -------------------------------------------------------------- PERMUTATION FEATURE IMPORTANCE (PFI) --------------------------------------------------------------



# ref : https://scikit-learn.org/1.5/modules/permutation_importance.html



class PFI_XAI_Methods:
    def __init__(self, model, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val if X_val is not None else X_train
        self.y_val = y_val if y_val is not None else y_train

        if feature_names is not None:
            self.feature_names = list(feature_names)
        elif isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
        else:
            self.feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]

    def compute_pfi(self, n_repeats=30 , random_state=42):
        """
        Compute Permutation Feature Importance (PFI).
        """
        result = permutation_importance(
            self.model, 
            self.X_val, 
            self.y_val, 
            n_repeats=n_repeats, 
            random_state= random_state 
        )

        importance_df = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance Mean": result.importances_mean,
            "Importance Std": result.importances_std
        }).sort_values(by="Importance Mean", ascending=False)

        return importance_df

    def save_importance(self, importance_df, file_path, file_format="csv"):
        """
        Save importance values to a file.
        """
        if file_format == "csv":
            importance_df.to_csv(file_path, index=False)
        elif file_format == "json":
            importance_df.to_json(file_path, orient="records", lines=True)
        else:
            raise ValueError("Unsupported file format. Use 'csv' or 'json'.")
        print(f"Saved feature importance to {file_path}")

    def plot_importance(self, importance_df, top_n=10):
        """
        Plot feature importance.
        """
        top_features = importance_df.head(top_n)
        plt.figure(figsize=(10, 6))
        plt.barh(top_features["Feature"], top_features["Importance Mean"], xerr=top_features["Importance Std"], color="skyblue")
        plt.xlabel("Mean Importance")
        plt.ylabel("Features")
        plt.title(f"Top {top_n} Permutation Feature Importance")
        plt.gca().invert_yaxis()
        plt.show()



# -------------------------------------------------------------- DALEX  --------------------------------------------------------------



class dalex_XAI_Methods:
    def __init__(self, model, X_train, y_train, feature_names=None):
        """
        Initialize the XAI_Methods class.

        :param model: Trained machine learning model.
        :param X_train: Training data used to compute feature importance.
        :param y_train: Training target values.
        :param feature_names: Optional list of feature names. If None, will use X_train columns.
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names if feature_names else X_train.columns

    def dalex_importance(self):
        """
        Compute DALEX feature importance using model_parts.

        :return: DataFrame of features and their importance values.
        """
        explainer = Explainer(self.model, self.X_train, self.y_train, label="Model Explainer")
        print("DALEX Explainer initialized.")

        # Compute feature importance using model_parts where  B = number of permutations
        feature_importance = explainer.model_parts(type='difference', B=10)  

        importance_df = feature_importance.result[['variable', 'dropout_loss']]
        importance_df.columns = ['Feature', 'Importance']
        importance_df = importance_df[importance_df['Feature'] != '_baseline_'].sort_values(by="Importance", ascending=False)

        return importance_df

    def save_importance(self, feature_importance, file_path, file_format="csv"):
        """
        Save feature importance to a file.

        :param feature_importance: DataFrame of features and their importance values.
        :param file_path: Path to save the file.
        :param file_format: Format to save the file ('csv' or 'json').
        """
        if file_format == "csv":
            feature_importance.to_csv(file_path, index=False)
        elif file_format == "json":
            feature_importance.to_json(file_path, orient="records", lines=True)
        else:
            raise ValueError("Unsupported file format. Use 'csv' or 'json'.")
        print(f"Feature importance saved to {file_path} in {file_format} format.")

    def plot_importance(self, feature_importance, top_n, title="Feature Importance - DALEX"):
            """
            Plot feature importance using a bar chart.

            :param feature_importance: DataFrame of features and their importance values.
            :param top_n: Number of top features to plot.
            :param title: Title of the plot.
            """
            top_features = feature_importance.head(top_n)

            plt.figure(figsize=(12, 8))
            sns.barplot(data=top_features, x="Importance", y="Feature", palette="viridis")
            plt.title(title, fontsize=16)
            plt.xlabel("Dropout Loss (Importance)", fontsize=14)
            plt.ylabel("Features", fontsize=14)
            plt.tight_layout()
            plt.show()


# -------------------------------------------------------------- PROFILED WEIGHTING (PROFWEIGHT) --------------------------------------------------------------



class profweight_XAI_Methods:
    def __init__(self, model, X_train, y_train, feature_names=None):
        """
        Initialize the ProfWeight XAI class.

        :param model: Trained machine learning model.
        :param X_train: Training data.
        :param y_train: Training labels.
        :param feature_names: Optional list of feature names. Defaults to X_train columns if None.
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names if feature_names else X_train.columns

    def compute_profweight(self, weighting_method="class_probabilities", metric="euclidean"):
        """
        Compute feature importance using ProfWeight.

        :param weighting_method: The profiling method used for weighting features.
                                 Options: 'class_probabilities', 'distances'.
        :param metric: Distance metric to use for distance-based weighting (if applicable).
        :return: DataFrame of feature importance values.
        """
        probabilities = self.model.predict_proba(self.X_train)

        if weighting_method == "class_probabilities":
            # Use class probabilities as weights
            weights = probabilities.max(axis=1)  # Take the max probability per sample
        elif weighting_method == "distances":
            # Use pairwise distances between samples as weights
            distances = pairwise_distances(self.X_train, metric=metric)
            weights = 1 / (1 + np.mean(distances, axis=1))  # Weight inversely proportional to distance
        else:
            raise ValueError("Unsupported weighting_method. Use 'class_probabilities' or 'distances'.")

        # Compute weighted feature contributions
        feature_contributions = np.abs(self.X_train.values) * weights[:, np.newaxis]
        feature_importance = np.mean(feature_contributions, axis=0)

        importance_df = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance": feature_importance
        }).sort_values(by="Importance", ascending=False)

        return importance_df

    def save_importance(self, feature_importance, file_path, file_format="csv"):
        """
        Save feature importance to a file.

        :param feature_importance: DataFrame of features and their importance values.
        :param file_path: Path to save the file.
        :param file_format: Format to save the file ('csv' or 'json').
        """
        if file_format == "csv":
            feature_importance.to_csv(file_path, index=False)
        elif file_format == "json":
            feature_importance.to_json(file_path, orient="records", lines=True)
        else:
            raise ValueError("Unsupported file format. Use 'csv' or 'json'.")
        print(f"Feature importance saved to {file_path} in {file_format} format.")

    def plot_top_features(self, feature_importance, top_n=25):
        """
        Plot the top N features by importance.

        :param feature_importance: DataFrame of features and their importance values.
        :param top_n: Number of top features to plot.
        """
        import matplotlib.pyplot as plt
        top_features = feature_importance.head(top_n)

        plt.figure(figsize=(10, 6))
        plt.barh(top_features["Feature"], top_features["Importance"], color="skyblue")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title(f"Top {top_n} Features by ProfWeight Importance")
        plt.gca().invert_yaxis()
        plt.show()
