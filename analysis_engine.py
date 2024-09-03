import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

class ProfoundAnalysisEngine:
    def __init__(self, data_file):
        """
        Initialize the engine with the provided data file.
        
        Args:
            data_file (str): Path to the CSV, JSON, or Excel file.
        """
        # Create output directory for analysis results
        self.output_dir = "analysis_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        if data_file.endswith('.csv'):
            self.df = pd.read_csv(data_file)
        elif data_file.endswith('.json'):
            self.df = pd.read_json(data_file)
        elif data_file.endswith('.xlsx'):
            self.df = pd.read_excel(data_file)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV, JSON, or Excel file.")
    
    def analyze(self):
        """
        Perform analysis on the dataset based on detected data types.
        """
        data_summary = self.df.describe(include='all')
        print("Data Summary:\n", data_summary)
    
        for column in self.df.columns:
            if self.df[column].dtype == 'object':
                self.handle_categorical_column(column)
            elif np.issubdtype(self.df[column].dtype, np.number):
                self.handle_numerical_column(column)
            else:
                print(f"Skipping analysis for column {column} due to unsupported data type.")
        
        self.perform_clustering()
        self.perform_pca()
        self.visualize()

        
    def handle_categorical_column(self, column):
        """
        Analyze a categorical column.
        
        Args:
            column (str): Name of the categorical column.
        """
        print(f"Analyzing categorical column: {column}")
        value_counts = self.df[column].value_counts()
        print(f"Value counts for {column}:\n", value_counts)
        
        # Encode the categorical column for further analysis
        label_encoder = LabelEncoder()
        self.df[column + '_encoded'] = label_encoder.fit_transform(self.df[column])
        
    def handle_numerical_column(self, column):
        """
        Analyze a numerical column.
        
        Args:
            column (str): Name of the numerical column.
        """
        print(f"Analyzing numerical column: {column}")
        
        # Check for NaN or inf and handle if present
        if self.df[column].isnull().any() or np.isinf(self.df[column]).any():
            print(f"Column {column} contains NaN or infinity. Handling these values.")
            self.df[column].replace([np.inf, -np.inf], np.nan, inplace=True)
            self.df[column].fillna(self.df[column].mean(), inplace=True)
            
        # Ensure we only use numerical columns for correlation
        numerical_columns = self.df.select_dtypes(include=np.number)
        correlation = numerical_columns.corr()
        
        print(f"Correlation matrix for numerical columns:\n{correlation}")
        
        # If there are multiple numerical columns, consider linear regression
        if len(numerical_columns.columns) > 1:
            self.perform_regression(column)

        
    def perform_regression(self, target_column):
        """
        Perform linear regression on the dataset to predict the target column.
        
        Args:
            target_column (str): The column to predict using linear regression.
        """
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]
        
        # Remove columns that are not features
        X = X.select_dtypes(include=[np.number])
        
        model = LinearRegression()
        model.fit(X, y)
        
        print(f"Regression coefficients for predicting {target_column}: {model.coef_}")
        print(f"Intercept: {model.intercept_}")
        print(f"R-squared: {model.score(X, y)}")
        
    def perform_clustering(self, n_clusters=3):
        """
        Perform clustering analysis to identify patterns in the data.
        
        Args:
            n_clusters (int): Number of clusters to form.
        """
        X = self.df.select_dtypes(include=np.number)
        
        if len(X.columns) > 1:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            self.df['Cluster'] = cluster_labels
            print(f"Cluster Centers:\n{kmeans.cluster_centers_}")
            print(f"Silhouette Score: {silhouette_score(X_scaled, cluster_labels)}")
        
            return cluster_labels, kmeans.cluster_centers_
    
    def perform_pca(self):
        """
        Perform Principal Component Analysis (PCA) for dimensionality reduction.
        """
        X = self.df.select_dtypes(include=np.number)
        if len(X.columns) > 2:
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(X)
            self.df['PCA1'] = pca_result[:, 0]
            self.df['PCA2'] = pca_result[:, 1]
            print(f"Explained Variance by PCA components: {pca.explained_variance_ratio_}")
    
    def visualize(self):
        """
        Generate visualizations for the analyzed data and save them as images.
        """
        sns.set(style="whitegrid")
        
        # Correlation heatmap
        numerical_cols = self.df.select_dtypes(include=np.number)
        if len(numerical_cols.columns) > 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(numerical_cols.corr(),
                        annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Correlation Heatmap')
            plt.savefig(f'{self.output_dir}/correlation_heatmap.png')
            plt.close()  # Close the figure to release memory

        # PCA Plot
        if 'PCA1' in self.df.columns and 'PCA2' in self.df.columns:
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=self.df, palette='viridis', s=100)
            plt.title('PCA Plot with Clustering')
            plt.savefig(f'{self.output_dir}/pca_plot.png')
            plt.close()

        # Categorical value counts
        for column in self.df.columns:
            if self.df[column].dtype == 'object':
                plt.figure(figsize=(10, 6))
                sns.countplot(y=column, data=self.df, palette='pastel')
                plt.title(f'Count Plot for {column}')
                plt.savefig(f'{self.output_dir}/count_plot_{column}.png')
                plt.close()
                
        # Linear regression results
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 1:
            for col in numeric_cols:
                if col != 'Cluster':  # Exclude Cluster label if present
                    plt.figure(figsize=(10, 6))
                    sns.regplot(x=col, y=self.df[numeric_cols[0]], data=self.df)
                    plt.title(f'Regression Plot for {col} vs {numeric_cols[0]}')
                    plt.savefig(f'{self.output_dir}/regression_plot_{col}.png')
                    plt.close()

file_path = 'olympics2024.csv'
engine = ProfoundAnalysisEngine(file_path)

engine.analyze()