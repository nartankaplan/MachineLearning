import pandas as pd

df = pd.read_csv("secondary_data.csv", sep=';')

categorical_columns = df.select_dtypes(include='object').columns

for column in categorical_columns:
    print(f"\n📁  categories in the '{column}' column and their frequencies:")
    print(df[column].value_counts())

