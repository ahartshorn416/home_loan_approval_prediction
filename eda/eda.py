
import pandas as pd
df = pd.read_csv("C:\\Users\\alica\\Downloads\\year_2023_loan_purposes_1.csv", nrows=5)
print(df.columns.tolist())
print(df.head())