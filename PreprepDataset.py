import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv("medlineplus_health_topic_details_summary_cleaned.csv")

# For classification: Pair each topic with its summary as the label
# Assuming the task is to classify queries (topics) into summaries
df['text'] = df['Topic']  # This will be the input text
df['label'] = df['Summary']  # This will be the target label

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Save the split datasets (optional, if you want to save the files)
train_df.to_csv("train_data.csv", index=False)
val_df.to_csv("val_data.csv", index=False)

# Display the first few rows of the prepared dataset
train_df.head()
