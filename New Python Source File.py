import pandas as pd

# 1. Load the CSV file
df = pd.read_csv('Verified_SAP_Shopify_Master.csv')

# 2. Clean or convert the size column to a sortable type if needed (e.g., numeric or string sorting)
# Assuming 'SAP SIZE' or 'SIZE' contains the size values. Let's use 'SAP SIZE' as seen in your sample.
# We will sort by GENERIC ARTICLE CODE and the size column to get the lowest size first.
df_sorted = df.sort_values(by=['GENERIC ARTICLE CODE', 'SAP SIZE'], ascending=[True, True])

# 3. Drop duplicates based on the parent SKU ('GENERIC ARTICLE CODE'), keeping the first occurrence (lowest size)
parent_skus_lowest_size = df_sorted.drop_duplicates(subset=['GENERIC ARTICLE CODE'], keep='first')

# 4. Save the result to a new CSV file
parent_skus_lowest_size.to_csv('parent_skus_lowest_size.csv', index=False)

print(f"Successfully extracted {len(parent_skus_lowest_size)} unique parent SKUs with their lowest size variant!")
print(parent_skus_lowest_size[['GENERIC ARTICLE CODE', 'GENERIC ARTICLE DESC', 'SAP SIZE', 'VARIANT ARTICLE CODE']].head())