import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the cleaned dataset
file_path = "medlineplus_health_topic_details_summary_cleaned.csv"  # Update with your actual file path
df_cleaned = pd.read_csv(file_path)

# Calculate the length of each summary
df_cleaned['Summary_Length'] = df_cleaned['Summary'].apply(len)

# Plot the distribution of summary lengths
plt.figure(figsize=(10, 6))
plt.hist(df_cleaned['Summary_Length'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Summary Lengths')
plt.xlabel('Length of Summary')
plt.ylabel('Number of Topics')
plt.show()

# Combine all summaries into a single string for word cloud generation
all_summaries = " ".join(df_cleaned['Summary'])

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_summaries)

# Display the wordcloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Common Keywords in Summaries')
plt.show()
