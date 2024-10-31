# CS414-Naive-Bayes
# READ ME 

# Overview
This project is part of the CS414 course, focusing on the implementation and evaluation of the Gaussian Naive Bayes algorithm for classification tasks. The project specifically targets Android malware detection using a dataset of app traffic data points. This repository provides the code to preprocess the data, handle class imbalance, and train a Naive Bayes classifier to classify benign and malicious Android applications.


# Project Structure

Dataset/: Contains the datasets used for training and testing. 
The main dataset, sampled_450.csv, consists of Android application data with multiple features for identifying malware types.
Naive_Bayes/: Contains the main Python script, naive_bayes_script.py, to run the Naive Bayes model.

# Requirements
To run the code, you will need the following Python libraries:

- pandas
- numpy
- scikit-learn
- imbalanced-learn
  
#Install the necessary libraries by running:

"pip install pandas numpy scikit-learn imbalanced-learn"

# Usage
Code Explanation
1. Data Loading: Loads the dataset from the specified directory and cleans the column names by stripping any leading or trailing whitespace.
2. Data Preprocessin: Drops the target label column and stores it in y_class, with features stored in X_class.
One-Hot Encoding: Encodes categorical features in X_class.
Imputation: Handles missing values using the mean strategy for numeric columns.
3. Label Encoding: Converts the target labels (e.g., benign and malware classes) into numerical format.
4. Train-Test Split: Splits the data into training and testing sets with an 80/20 split.
5. Class Imbalance Handling: Applies SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes in the training data, ensuring the model is trained on a balanced dataset.
6. Model Training: Initializes and trains a Gaussian Naive Bayes classifier on the balanced training data.
7.Model Evaluation:
-Predicts the labels for the test set.
-Prints the accuracy score and classification report (precision, recall, F1-score for each class).


#Running the Code
To run the script, navigate to the Naive_Bayes folder and execute the Python script:
#"python naive_bayes_script.py"

#Results
The results include:
Accuracy Score: Provides the overall accuracy of the model on the test set.
Classification Report: Displays precision, recall, and F1-score for each class, helping to understand the model's performance on different types of malware and benign cases.

#License
This project is licensed under the MIT License.
