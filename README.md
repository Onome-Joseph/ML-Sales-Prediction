# Sales Prediction Model
This repository contains a **Sales Prediction** Machine Learning Model built to predict sales for a given dataset of  features containing different goods/services of a business and their attributes.
#  Aim of the Project
The primary goal of this project is to develop a machine learning model capable of accurately predicting sales. By leveraging advanced regression techniques and feature engineering, the model empowers businesses to:
- Optimize their sales strategies.
- Anticipate revenue.
- Make data-backed decisions to maximize profitability.
# Algorithms Used:
- XGBoost Regression: A powerful gradient boosting technique for accurate predictions.
- Polynomial Features Transformation: For capturing non-linear relationships in the data.
# Applications
- Inventory Management: Prevents overstocking or understocking by predicting demand accurately.
- Marketing Optimization: Assists in targeting campaigns based on predicted sales spikes or drops.
- Resource Allocation: Helps allocate resources to maximize profit based on expected sales patterns.

# Running the Sales Prediction FastAPI App Locally
---
![Screenshot of the Sales Prediction Model](https://github.com/Onome-Joseph/ML-Sales-Prediction/blob/main/Screenshot.jpeg)
## Steps to Run the App
### 1. Clone the Repository

```bash
git clone https://github.com/Onome-Joseph/ML-Sales-Prediction.git
```
  ```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the FastAPI App
```bash
uvicorn FlaskAPI_(with frontend):app --reload
```
### 4. Access the API

Once the server is running, you can access the API documentation in your web browser at:
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
---
### Contributions
Feel free to fork the repository and raise pull requests for enhancements or bug fixes.
