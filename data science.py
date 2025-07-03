import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
df = pd.read_csv("all_recs.csv")

# Show first 5 rows
print("First 5 rows:")
print(df.head())

# Show columns
print("\nColumns in the file:")
print(df.columns)

# Show missing values (if any)
print("\nMissing values:")
print(df.isnull().sum())

# Summary statistics
print("\nSummary statistics:")
print(df.describe(include='all'))

# Number of times each person was recognized
if 'Name' in df.columns:
    print("\nNumber of times each person was recognized:")
    print(df['Name'].value_counts())

    # Plot frequency of recognition
    sns.countplot(data=df, x='Name', order=df['Name'].value_counts().index)
    plt.title("Face Recognition Frequency")
    plt.ylabel("Number of Recognitions")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Distribution over time (by date)
if 'Time' in df.columns:
    df['Date'] = pd.to_datetime(df['Time']).dt.date
    date_counts = df['Date'].value_counts().sort_index()

    print("\nRecognition counts by date:")
    print(date_counts)

    # Plot recognitions over time
    date_counts.plot(kind='bar')
    plt.title("Face Recognitions Over Time")
    plt.xlabel("Date")
    plt.ylabel("Recognitions")
    plt.tight_layout()
    plt.show()