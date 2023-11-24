import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Load the dataset
file_path = 'C:/datasetREDI/news_dataset.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Remove duplicates
data.drop_duplicates(inplace=True)


# Save the cleaned dataset to a new file (optional)
cleaned_file_path = 'C:/datasetREDI/news_dataset_cleaned.csv'
data.to_csv(cleaned_file_path, index=False)

# Display information about the cleaned dataset
print("Cleaning process completed!")
print("Shape of cleaned dataset:", data.shape)

print(data.columns)

# Example: Countplot for the 'Sport' column
plt.figure(figsize=(8, 6))
sns.countplot(data['Sport'])
plt.title('Count of Sport Categories')
plt.xlabel('Sport')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()

# Convert 'Sport' column values to lowercase for case-insensitive comparison
data['Sport'] = data['Sport'].str.lower()

# Filter the dataset for specific sports: football, basketball, tennis, and mma
selected_sports = ['football', 'basketball', 'tennis', 'mma']
filtered_data = data[data['Sport'].isin(selected_sports)]

# Countplot to compare frequencies of news for the selected sports
plt.figure(figsize=(10, 6))
sns.countplot(data=filtered_data, x='Sport', order=selected_sports)
plt.title('Comparison of News Frequency for Football, Basketball, Tennis, and MMA')
plt.xlabel('Sport')
plt.ylabel('Count of News')
plt.show()


print(filtered_data.head())

# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Filter the dataset for specific sports: football, basketball, tennis, and mma
selected_sports = ['football', 'basketball', 'tennis', 'mma']
filtered_data = data[data['Sport'].str.lower().isin(selected_sports)]

# Get the start and end dates for the last month
end_date = pd.Timestamp.now().normalize()
start_date = end_date - pd.offsets.MonthBegin(1)

# Filter the dataset for the last month
last_month_data = filtered_data[(filtered_data['Date'] >= start_date) & (filtered_data['Date'] <= end_date)]

# Group by date and sport to count occurrences
daily_counts = last_month_data.groupby([last_month_data['Date'].dt.date, 'Sport']).size().unstack(fill_value=0)

# Plotting
plt.figure(figsize=(12, 6))
sns.lineplot(data=daily_counts)
plt.title('News Frequency for Football, Basketball, Tennis, and MMA (Last Month)')
plt.xlabel('Date')
plt.ylabel('Count of News')
plt.legend(title='Sport', loc='upper right')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Group by sport to count occurrences
sport_counts = last_month_data['Sport'].value_counts()

# Create a polar plot
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

# Compute angles and heights for radar chart
angles = [n / float(len(sport_counts)) * 2 * 3.14159 for n in range(len(sport_counts))]
counts = sport_counts.tolist()
counts += [counts[0]]
angles += [angles[0]]

# Plotting
plt.xticks(angles[:-1], sport_counts.index, color='black', size=10)
ax.plot(angles, counts, linewidth=1, linestyle='solid')
ax.fill(angles, counts, 'blue', alpha=0.1)

plt.title('News Frequency for Football, Basketball, Tennis, and MMA (Last Month)')
plt.show()

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(sport_counts, labels=sport_counts.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('News Frequency for Football, Basketball, Tennis, and MMA (Last Month)')
plt.show()

#ML

# Assuming you have a DataFrame called 'filtered_data' with a 'Sport' column

# Create a function to categorize sports based on ball-related keywords
def categorize_sports(sport_name):
    ball_keywords = ['football', 'soccer', 'basketball', 'volleyball', 'rugby', 'handball', 'tennis']
    for keyword in ball_keywords:
        if keyword in sport_name.lower():
            return 'Ball Sports'
    return 'Non-Ball Sports'

# Create a copy of the DataFrame to avoid warnings
filtered_data_copy = filtered_data.copy()

# Apply the categorization function to create a new column
filtered_data_copy['Sport_Category'] = filtered_data_copy['Sport'].apply(categorize_sports)

# Prepare data for clustering
text_representation = filtered_data_copy['Sport_Category']

# Convert text data to numerical format using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(text_representation)

# Apply KMeans clustering with explicit parameter settings
kmeans = KMeans(n_clusters=2, n_init=10)  # Two clusters: Ball Sports and Non-Ball Sports
kmeans.fit(X)

# Add cluster labels to the DataFrame using .loc to avoid warnings
filtered_data_copy.loc[:, 'Cluster'] = kmeans.labels_

# Display the clustered data
print(filtered_data_copy[['Sport', 'Sport_Category', 'Cluster']])

# Visualize the distribution of clustered sports categories as a pie chart
cluster_counts = filtered_data_copy['Cluster'].value_counts()
labels = ['Cluster 0, Ball Sports', 'Cluster 1, Non-Ball Sports']  # Assuming you have two clusters

plt.figure(figsize=(8, 6))
plt.pie(cluster_counts, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Clusters')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

