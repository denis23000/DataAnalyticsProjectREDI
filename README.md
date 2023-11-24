# DataAnalyticsProjectREDI
# Sports News Analysis
Description
This Python project performs data analysis on a sports news dataset utilizing Pandas, Matplotlib, Seaborn, scikit-learn, and TF-IDF vectorization. The analysis includes data cleaning, exploratory data analysis (EDA), visualizations, and machine learning tasks such as clustering.
Tasks and Solutions
## 1. Data Cleaning and Preparation
Task: Load and clean the dataset to ensure data integrity.
Solution: Utilized Pandas to load the dataset, check for missing values, remove duplicates, and handle missing values if necessary.
## 2. Exploratory Data Analysis (EDA)
Task: Visualize sports news data distributions.
Solution: Employed Matplotlib and Seaborn to create count plots, histograms, and line plots to explore news frequency and trends across sports categories.
## 3. Predictive Analysis - Machine Learning (ML) 
### 3.1 Predicting Sports News Frequency
Task: Predict the frequency of specific sports news (football, basketball, tennis, MMA) for the last month.
Solution: Utilized Pandas to filter data for the last month and applied time-based analysis to predict news frequency.
### 3.2 Clustering Sports Categories
Task: Group sports categories based on ball-related keywords.
Solution: Employed TF-IDF vectorization and KMeans clustering from scikit-learn to categorize sports into 'Ball Sports' and 'Non-Ball Sports'.
Visualizations
Task: Visualize clustering results and sports category distributions.
Solution: Utilized Matplotlib to create pie charts, line plots, and polar plots displaying news frequency distributions and clustered sports categories.
Machine Learning Results
Task: Display clustered sports data.
Solution: Applied KMeans clustering and displayed the clustered sports data alongside a pie chart showcasing the distribution of clustered categories.

Dataset download link: https://www.kaggle.com/datasets/shivamtaneja2304/google-news-sports/
