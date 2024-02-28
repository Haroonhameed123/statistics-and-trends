import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis, skew

def load_and_clean_data(filepath):
    """
    Load and perform initial cleaning on the data scientist salaries dataset.
    Parameters:
    - filepath: str, path to the CSV file containing the dataset.
    Returns:
    - DataFrame, cleaned dataset.
    """
    data = pd.read_csv(filepath)
    return data

def plot_salary_distribution(data):
    """
    Plot the distribution of salaries in USD.
    Parameters:
    - data: DataFrame, dataset containing salary information.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data['salary_in_usd'], bins=30, color='#ff6361', kde=True, edgecolor='black')
    plt.title('Distribution of Salaries in USD', fontsize=16)
    plt.xlabel('Salary in USD', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.show()

def plot_salary_by_experience(data):
    """
    Plot average salary by experience level.
    Parameters:
    - data: DataFrame, dataset containing salary and experience level information.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x='experience_level', y='salary_in_usd', data=data, palette='Spectral')
    plt.title('Average Salary by Experience Level', fontsize=16)
    plt.xlabel('Experience Level', fontsize=14)
    plt.ylabel('Average Salary in USD', fontsize=14)
    plt.show()

def plot_salary_by_job_title(data):
    """
    Plot salary range by job title focusing on common job titles to reduce overcrowding.
    Parameters:
    - data: DataFrame, dataset containing salary and job title information.
    """
    common_titles = data['job_title'].value_counts().head(5).index
    filtered_data = data[data['job_title'].isin(common_titles)]
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='salary_in_usd', y='job_title', data=filtered_data, palette='Set3')
    plt.title('Salary Range for Common Data Science Job Titles', fontsize=16)
    plt.xlabel('Salary in USD', fontsize=14)
    plt.ylabel('Job Title', fontsize=14)
    plt.show()

def plot_work_model_impact(data):
    """
    Plot the impact of work model on salary.
    Parameters:
    - data: DataFrame, dataset containing salary and work model information.
    """
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='work_models', y='salary_in_usd', data=data, palette='cool')
    plt.title('Impact of Work Model on Salary in Data Science Roles', fontsize=16)
    plt.xlabel('Work Model', fontsize=14)
    plt.ylabel('Salary in USD', fontsize=14)
    plt.show()

def perform_statistical_analysis(data):
    """
    Perform and print statistical analysis including describe, correlation, kurtosis, and skewness on numeric columns.
    Parameters:
    - data: DataFrame, dataset to analyze.
    """
    numeric_data = data.select_dtypes(include=[np.number])
    print(numeric_data.describe())
    print("\nCorrelation Matrix:\n", numeric_data.corr())
    print("\nKurtosis:\n", numeric_data.apply(lambda x: kurtosis(x) if np.issubdtype(x.dtype, np.number) else np.nan))
    print("\nSkewness:\n", numeric_data.apply(lambda x: skew(x) if np.issubdtype(x.dtype, np.number) else np.nan))

def plot_salary_over_time_adjusted(data, jitter=0.2, alpha=0.7):
    """
    Plot an adjusted scatter chart showing the relationship between salary in USD and work year,
    with improvements for clarity.
    
    Parameters:
    - data: DataFrame, dataset containing salary and work year information.
    - jitter: float, amount of jitter to add to the 'work_year' axis.
    - alpha: float, transparency of points.
    """
    plt.figure(figsize=(12, 7))
    # Adding jitter on the 'work_year' and adjusting the transparency of points
    sns.stripplot(x='work_year', y='salary_in_usd', data=data, hue='experience_level', 
                    palette='viridis', alpha=alpha, jitter=jitter, dodge=True)
    
    plt.title('Salary in USD Over Time by Experience Level and Work Model', fontsize=16)
    plt.xlabel('Work Year', fontsize=14)
    plt.ylabel('Salary in USD', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)  # Rotate x-axis labels for better readability
    plt.yticks(fontsize=12)
    plt.legend(title='Experience Level', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', title_fontsize='13')
    plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')  # Enhance grid visibility
    plt.tight_layout()  # Adjust layout to make room for the legend
    plt.show()

# Example usage (assuming the rest of the functions and data loading are defined as before):
if __name__ == "__main__":
    filepath = 'data_science_salaries.csv'
    salaries_data = load_and_clean_data(filepath)
    perform_statistical_analysis(salaries_data)
    plot_salary_distribution(salaries_data)
    plot_salary_by_experience(salaries_data)
    plot_salary_by_job_title(salaries_data)
    plot_work_model_impact(salaries_data)
    plot_salary_over_time_adjusted(salaries_data, jitter=0.2, alpha=0.7)