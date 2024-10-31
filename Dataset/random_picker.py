import pandas as pd

# Load the original CSV file
file_path = 'Android_Malware7features.csv'  # Replace with your file's name/path
df  = pd.read_csv(file_path)

# Sample 100 rows from each category in the 'label' column
sampled_df = df.groupby('Label').apply(lambda x: x.sample(n=100, random_state=42)).reset_index(drop=True)

# Save the sampled data to a new CSV file
sampled_df.to_csv('sampled_400.csv', index=False)

print("Sampled 400 rows (100 from each category) saved to 'sampled_400.csv'.")
