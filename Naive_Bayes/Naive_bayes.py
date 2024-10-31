import pandas as pd
from os import path
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

PROJECT_ROOT = path.abspath(path.dirname(path.dirname(__file__)))
DATA_DIR = path.join(PROJECT_ROOT, "Dataset")
DATA_FILE = path.join(DATA_DIR, "sampled_450.csv")

# Load the dataset
df = pd.read_csv(DATA_FILE, low_memory=False)

# Clean column names
df.columns = df.columns.str.strip()
# Define selected features and target label
target_label = 'Label'
# Prepare data for classification
X_class = df.drop(columns=[target_label]) 
y_class = df[target_label]

# Encode categorical features first
X_class_encoded = pd.get_dummies(X_class, drop_first=True)

# Impute missing values using mean strategy for numeric columns
imputer = SimpleImputer(strategy='mean')
X_class_imputed = imputer.fit_transform(X_class_encoded)

# Encode target labels
label_encoder = LabelEncoder()
y_class_encoded = label_encoder.fit_transform(y_class)

# Split the data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X_class_imputed, y_class_encoded, test_size=0.2, random_state=90, stratify=y_class_encoded)

# Handle class imbalance with SMOTE
smote = SMOTE()
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Initialize the Gaussian Naive Bayes classifier
gnb = GaussianNB()

# Train the model
gnb.fit(X_train_balanced, y_train_balanced)

# Make predictions
y_pred = gnb.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

# Print evaluation results
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)