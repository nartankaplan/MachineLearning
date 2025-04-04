import pandas as pd

df = pd.read_csv("secondary_data.csv", sep=';')

print("🧾 Columns and their data types:\n")
print(df.dtypes)

numeric_columns_to_convert = ['cap-diameter', 'stem-height', 'stem-width']
for col in numeric_columns_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')

numerical_columns = df.select_dtypes(include='number').columns

for column in numerical_columns:
    print(f"\n📊 {column} Summary:")
    print(f"Mean: {df[column].mean():.2f}")
    print(f"Median: {df[column].median():.2f}")
    print(f"Standard Deviation: {df[column].std():.2f}")
    print(f"Min: {df[column].min():.2f}")
    print(f"25% (1st Quartile): {df[column].quantile(0.25):.2f}")
    print(f"50% (Median): {df[column].quantile(0.50):.2f}")
    print(f"75% (3rd Quartile): {df[column].quantile(0.75):.2f}")
    print(f"Max: {df[column].max():.2f}")


