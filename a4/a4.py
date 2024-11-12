import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('California_Houses.csv')

#adding a new column for model B 
data['min_distance_to_city'] = data[['Distance_to_LA', 'Distance_to_SanDiego', 
                                     'Distance_to_SanJose', 'Distance_to_SanFrancisco']].min(axis=1)

# Model A features: each distance to a big city as a separate feature
features_A = data[['Median_Income', 'Median_Age', 'Tot_Rooms', 'Tot_Bedrooms', 
                   'Population', 'Households', 'Distance_to_LA', 
                   'Distance_to_SanDiego', 'Distance_to_SanJose', 'Distance_to_SanFrancisco']]

# Model B features: single minimum distance to nearest big city
features_B = data[['Median_Income', 'Median_Age', 'Tot_Rooms', 'Tot_Bedrooms', 
                   'Population', 'Households', 'min_distance_to_city']]

#what we want to predict, in this case median house value
target = data['Median_House_Value']

# Split the data into training and testing sets
# about 80% of the data set is training while 0.2 or 20% is the tessting set
# random_state is the rng seed used for the random data splitting for training
X_train_A, X_test_A, y_train, y_test = train_test_split(features_A, target, test_size=0.2, random_state=42)
X_train_B, X_test_B, _, _ = train_test_split(features_B, target, test_size=0.2, random_state=42)

#train the models
# Model A
model_A = LinearRegression()
model_A.fit(X_train_A, y_train)

# Model B
model_B = LinearRegression()
model_B.fit(X_train_B, y_train)

# Use metrics like Mean Squared Error (MSE) and R^2 score to compare the performance of both models.
# for step 7 inerpret the better model by testing which one predicts the best value#
pred_A = model_A.predict(X_test_A)
pred_B = model_B.predict(X_test_B)

mse_A = mean_squared_error(y_test, pred_A)
r2_A = r2_score(y_test, pred_A)

mse_B = mean_squared_error(y_test, pred_B)
r2_B = r2_score(y_test, pred_B)

print("Model A Mean Squared Error:", mse_A)
print("Model A R² Score:", r2_A)
print("\nModel B Mean Squared Error:", mse_B)
print("Model B R² Score:", r2_B)

if r2_A > r2_B:
    print("\nModel A has a higher R² score than Model B and performs better.")
elif r2_B > r2_A:
    print("\nModel B has a higher R² score than Model A and performs better.")
else:
    print("\nBoth models have similar performance.")
    
#Presentation Example Usage
compare_2_homes = pd.DataFrame({
    'Median Income': [5.5, 6.8],
    'Median Age': [15, 10],
    'Total Rooms': [800, 1200],
    'Total Bedrooms': [150, 300],
    'Population': [300, 500],
    'Households': [100, 200],
    'Distance to Los Angeles': [10000, 50000],
    'Distance to San Diego': [50000, 10000],
    'Distance to San Jose': [450000, 400000],
    'Distance to San Francisco': [500000, 600000] 
})

compare_2_homes.columns = ['Median_Income', 'Median_Age', 'Tot_Rooms', 'Tot_Bedrooms', 
'Population', 'Households', 'Distance_to_LA', 'Distance_to_SanDiego', 'Distance_to_SanJose', 'Distance_to_SanFrancisco']

# Add the min_distance_to_city column for Model B
compare_2_homes['min_distance_to_city'] = compare_2_homes[['Distance_to_LA', 'Distance_to_SanDiego', 
'Distance_to_SanJose', 'Distance_to_SanFrancisco']].min(axis=1)

predictionB = model_B.predict(compare_2_homes[features_B.columns])

print() # format line

# Display predictions
for i, price in enumerate(predictionB, 1):
    print(f"Predicted Price for Property {i}: ${price:,.2f}")

print() #format line

# Scatter plot of Predicted vs Actual House Values for Model A
plt.figure(figsize=(8, 6))
plt.scatter(y_test, pred_B, alpha=0.5, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Ideal Fit Line")
plt.title("Model B: Predicted vs Actual House Values")
plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.legend()
plt.show()
