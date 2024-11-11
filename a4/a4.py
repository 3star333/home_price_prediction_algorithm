import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

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
from sklearn.model_selection import train_test_split
X_train_A, X_test_A, y_train, y_test = train_test_split(features_A, target, test_size=0.2, random_state=42)
X_train_B, X_test_B, _, _ = train_test_split(features_B, target, test_size=0.2, random_state=42)

#train the models
# Model A
model_A = LinearRegression()
model_A.fit(X_train_A, y_train)

# Model B
model_B = LinearRegression()
model_B.fit(X_train_B, y_train)

#step 6-7 to do,
# Use metrics like Mean Squared Error (MSE) and R² score to compare the performance of both models.
# for step 7 inerpret the better model by testing which one predicts the best value#


#program example usage (no userinput yet)
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
#make predictions here by calling the prediction function buliltin
# model_N.predict(compare_2_homes)

#print the results 
# Display predictions using loop
# for i, price in enumerate(predicted_price, 1):
#    print(f"Predicted Price for Property {i}: ${price:,.2f}")