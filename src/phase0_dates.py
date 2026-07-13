import pandas as pd
df = pd.read_csv(r'd:\Analytics projects\project5\AgriPriceIndia\data\raw\Agriculture_price_dataset.csv', low_memory=False)
df['Price Date'] = pd.to_datetime(df['Price Date'], format='%m/%d/%Y')
for crop in sorted(df['Commodity'].unique()):
    s = df[df['Commodity'] == crop]
    yr_counts = s['Price Date'].dt.year.value_counts().sort_index().to_dict()
    mn = s['Price Date'].min().date()
    mx = s['Price Date'].max().date()
    n_days = s['Price Date'].dt.date.nunique()
    print(f"{crop}: {mn} to {mx} ({n_days} unique days), rows by year: {yr_counts}")
