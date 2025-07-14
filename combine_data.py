import pandas as pd

# Load the CSV files
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

# Add labels
fake_df["label"] = "FAKE"
true_df["label"] = "REAL"

# Combine the datasets
combined_df = pd.concat([fake_df, true_df], ignore_index=True)

# Keep only text and label columns
combined_df = combined_df[["text", "label"]]

# Save to new file
combined_df.to_csv("fake_news.csv", index=False)

print("✅ fake_news.csv created successfully!")
