# 🏡 House Price Prediction   

the api link :
# ------------------------------------------------------

# https://ml-house-price-prediction-model-3.onrender.com/

# ----------------------------------------------------------

This project predicts house prices using a **Random Forest Regression** model. The workflow includes **data preprocessing, visualization, model training, hyperparameter tuning, and evaluation**.  

## 📌 Project Overview  

The dataset (`house-price.csv`) contains **545 rows and 13 columns**, with features such as:  
- **Numerical:** Area (sq ft), bedrooms, bathrooms, stories, parking, etc.  
- **Categorical:** Main road access, guest room, basement, air conditioning, furnishing status, etc.  

The goal is to **train a robust model** that accurately predicts house prices based on these features.  

## 🛠️ Project Workflow  

1. **Data Loading & Exploration** – Import the dataset, check structure, and identify key features.  
2. **Preprocessing & Feature Engineering** – Handle missing values, outliers, and encode categorical variables.  
3. **Data Visualization & Analysis** – Use scatter plots, box plots, and correlation heatmaps to explore relationships.  
4. **Model Implementation & Training** – Split data, preprocess features, and train a **Random Forest Regressor**.  
5. **Hyperparameter Tuning** – Optimize the model using **Grid Search** for the best performance.  
6. **Model Evaluation** – Assess performance using **MAE, MSE, RMSE, and R² metrics**.  

## 📂 Key Files  

- 📜 **`housePrice.ipynb`** – Jupyter Notebook containing the full implementation.  
- 📊 **`house-price.csv`** – The dataset used for training and testing.  

## 🚀 How to Run  

1. **Install dependencies**:  
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn

2. Open the Jupyter Notebook (**housePrice.ipynb**) in Jupyter Lab or Notebook.

3. Run each cell sequentially to execute the code.

4. View results, including visualizations and model evaluation metrics.   
