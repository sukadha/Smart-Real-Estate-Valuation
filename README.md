# 🏠 House Price Prediction with AI Assistant

A machine learning-based web application that predicts house prices using property details such as area, number of bedrooms, bathrooms, main road availability, and basement availability.

The project combines a **Machine Learning prediction model** with a **Gemini AI Assistant**, allowing users to make predictions and ask questions about the housing dataset through a conversational interface.

---

## 📌 Project Overview

The **House Price Prediction** system uses a Gradient Boosting Regression model to estimate the price of a house based on user-provided property details.

The application has two main components:

1. **House Price Prediction**
   - Users enter house details.
   - The Flask backend sends the input to the trained ML model.
   - The model predicts the estimated house price.

2. **Gemini AI Assistant**
   - Users can ask questions in natural language.
   - The AI Assistant can understand house-related queries.
   - It can provide ML-based price predictions when sufficient information is available.
   - It can answer questions related to the housing dataset.

---

## 🎯 Aim

To develop an AI-powered web application that predicts house prices using machine learning and provides an interactive Gemini AI Assistant for answering housing and dataset-related questions.

---

## 🎯 Objectives

- Build a machine learning model for house price prediction.
- Use important housing features to estimate prices.
- Develop a Flask backend for ML prediction APIs.
- Develop a React/Vite frontend for an interactive user interface.
- Integrate Gemini AI for conversational assistance.
- Allow users to ask house-price questions using natural language.
- Provide dataset-related information through the AI Assistant.
- Create a simple and user-friendly prediction interface.

---

## 🧠 Machine Learning Model

The project uses a:

### Gradient Boosting Regressor

The model is trained using the following features:

| Feature | Description |
|---|---|
| `area` | Area of the house in square feet |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `mainroad` | Whether the house is connected to the main road |
| `basement` | Whether the house has a basement |

### Target Variable

```text
price
