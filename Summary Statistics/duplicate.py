import pandas as pd

# CSV reading
df = pd.read_csv("secondary_data.csv")

# Finding duplicate entries
duplicates = df[df.duplicated()]

# Printing duplicate entries
print("🔁 Duplicate entries: ")
print(duplicates)

# Duplicate sayısını yazdır
duplicate_count = df.duplicated().sum()
print(f"\n🔢 Total {duplicate_count} duplicate Entries found.")
