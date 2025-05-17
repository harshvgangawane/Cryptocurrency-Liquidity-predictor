import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlations(df, cols=None):
    if cols is None:
        cols = df.select_dtypes(include='number').columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()

def plot_feature_importances(model, feature_names, n=10):
    importances = model.feature_importances_
    indices = importances.argsort()[::-1][:n]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices])
    plt.title('Top Feature Importances')
    plt.tight_layout()
    plt.show()