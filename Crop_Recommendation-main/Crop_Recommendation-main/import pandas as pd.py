import pandas as pd

# Load dataset
file_path = "C:/Users/Manoj/Downloads/Crop_Recommendation-main/Crop_Recommendation-main/dataset.csv"
df = pd.read_csv(file_path)

# Print column names
print("✅ Columns in dataset.csv:", df.columns)
