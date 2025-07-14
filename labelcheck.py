import pandas as pd
df = pd.read_csv("fake_news.csv")
print("Unique labels in your dataset:")
print(df['label'].unique())