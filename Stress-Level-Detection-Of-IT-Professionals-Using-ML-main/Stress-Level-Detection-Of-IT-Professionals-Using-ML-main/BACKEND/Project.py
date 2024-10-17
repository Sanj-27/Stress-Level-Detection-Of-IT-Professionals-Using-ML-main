import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load the data using pandas
df = pd.read_csv('C:\Users\Varun\Downloads\Stress-Level-Detection-Of-IT-Professionals-Using-ML-main\Stress-Level-Detection-Of-IT-Professionals-Using-ML-main\stress_detection_IT_professionals_dataset.csv')
# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Display the last few rows of the dataset
print("\nLast few rows of the dataset:")
print(df.tail())

# Display the shape of the dataset
print("\nShape of the dataset:")
print(df.shape)

# Display the data types of each column
print("\nData types of each column:")
print(df.dtypes)

# Display the number of missing values in each column
print("\nMissing values in each column:")
print(df.isnull().sum())

# Display dataset information including number of columns, rows, data types, null counts, and memory usage
print("\nDataset information:")
df.info()

# Display unique values in each column
print("\nUnique values in each column:")
print(df.nunique())

# Display statistical information about the dataset
print("\nStatistical information about the dataset:")
print(df.describe(include='all').T)

# Display the number of duplicated rows
print("\nNumber of duplicated rows:")
print(df.duplicated().sum())

# Display missing values again for confirmation
print("\nMissing values in each column (again):")
print(df.isnull().sum())
