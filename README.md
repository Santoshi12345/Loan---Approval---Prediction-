# Loan Approval Prediction using Artificial Intelligence

## 📌 Project Overview
This project predicts whether a loan application will be Approved or Rejected using a Machine Learning model.

The system uses a Random Forest Classifier to analyze applicant details such as income, education, credit history, property area, and other factors to determine loan approval status.

This project was developed as part of an Artificial Intelligence Internship.

---

## 📂 Dataset Information

The dataset contains the following features:

- Loan_ID – Unique loan identifier
- Gender – Male / Female
- Married – Yes / No
- Dependents – Number of dependents (0,1,2,3+)
- Education – Graduate / Not Graduate
- Self_Employed – Yes / No
- ApplicantIncome – Applicant’s monthly income
- CoapplicantIncome – Co-applicant’s monthly income
- LoanAmount – Loan amount requested
- Credit_History – 1 (Good) / 0 (Bad)
- Property_Area – Urban / Semiurban / Rural
- Loan_Status – Target variable (Approved / Rejected)

---

## ⚙️ Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn
- Random Forest Algorithm

---

## 🧠 Machine Learning Approach

### 1️⃣ Data Preprocessing
- Removed Loan_ID column (not useful for prediction)
- Converted categorical variables into numeric using Label Encoding
- Split dataset into training and testing sets (80% training, 20% testing)

### 2️⃣ Model Used
Random Forest Classifier with:
- 200 decision trees
- Fixed random_state for reproducibility

Random Forest was selected because:
- It reduces overfitting
- Works well with mixed data types
- Provides feature importance scores
- Delivers high classification accuracy

---

## 📊 Model Evaluation

The model performance was evaluated using:

- Accuracy Score
- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix

Feature Importance was also calculated to understand which factors influence loan approval the most.

---

## 📈 Output Includes

- Model Accuracy
- Detailed Classification Report
- Confusion Matrix
- Feature Importance Table
- Feature Importance Visualization Graph

---

## ▶️ How to Run the Project

1. Install required libraries:

   pip install pandas matplotlib scikit-learn

2. Make sure the dataset file is named:

   loan_dataset.csv

3. Run the Python script:

   python loan_prediction.py

---

## 🔮 Future Improvements

- Hyperparameter tuning
- Model comparison (Decision Tree vs Random Forest)
- Add cross-validation
- Deploy as a web application
- Use larger real-world dataset

---

## 👩‍💻 Author

Artificial Intelligence Internship Project  
Loan Approval Prediction System
