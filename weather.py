import pandas as pd
from sklearn import svm
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
#Applying a Weather Kaggle dataset to perform classification and predict Cloud Cover (Clear, Partly Cloudy, Cloudy, or Overcast)

# Using Pandas to read the kaggle .csv file and map input and output for the model

data = pd.read_csv("weather_classification_data.csv")

# x = data[["Humidity", "Precipitation (%)", "Visibility (km)"]]
x = data[["Humidity", "Precipitation (%)", "Visibility (km)", "Temperature", "UV Index"]]
y = data[["Cloud Cover"]]

# Training model using SVC
model = svm.SVC()
model.fit(x, y)

# Making Predictions using the data
print("Classification Predictions")
# Clear
clear_example1 = [35, 10, 9.0, 32.0, 9]   # row from dataset
clear_example2 = [25,  6, 8.0, 35.0, 10]   # row from dataset
print("Predicted cloud cover for clear_example1:", model.predict([clear_example1])[0])
print("Predicted cloud cover for clear_example2:", model.predict([clear_example2])[0])

# Partly Cloudy
partly_cloudy_example1 = [60, 25, 8.0, 22.0, 5]
partly_cloudy_example2 = [70, 40, 7.0, 26.0, 4]
print("Predicted cloud cover for partly_cloudy_example1:", model.predict([partly_cloudy_example1])[0])
print("Predicted cloud cover for partly_cloudy_example2:", model.predict([partly_cloudy_example2])[0])

# Cloudy (For some reason cloudy never appears as a prediction)
cloudy_example1 = [66, 28, 5.0, 19.0, 3]
cloudy_example2 = [72, 34, 6.0, 17.0, 2]
print("Predicted cloud cover for cloudy_example1:", model.predict([cloudy_example1])[0])
print("Predicted cloud cover for cloudy_example2:", model.predict([cloudy_example2])[0])

# Overcast
overcast_example1 = [90, 85, 2.0, 16.0, 1]
overcast_example2 = [95, 95, 3.0, 12.0, 0]
print("Predicted cloud cover for overcast_example1:", model.predict([overcast_example1])[0])
print("Predicted cloud cover for overcast_example2:", model.predict([overcast_example2])[0])

print("------------------------------------\nEnd of Classification predictions\n")

# Applying Regression Analysis to this cloud cover example. I'll have to map each category of cloud cover to a percentage in order to produce a numerical prediction
# So now, Clear = 0 - 29.9%, Partly Cloudy = 30 - 59.9%, Cloudy = 60 - 89.9%, and Overcast = 90 - 100%
cloudiness_scale = {
    "clear": 0,
    "partly cloudy": 30,
    "cloudy": 60,
    "overcast": 90
}
# This maps the data within the columns to this new variable with the integer values instead.
data["Cloudiness (%)"] = data["Cloud Cover"].str.lower().map(cloudiness_scale)

# This is the y that we'll train our polynomial regression model on
y_pr = data["Cloudiness (%)"]

pr_model = make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression())
pr_model.fit(x, y_pr)

print("Polynomial Regression cloudiness percentage prediction")
#Testing out the polynomial regression model on our previous samples from the classification attempt
print("Predicted cloud percentage for clear_example1:", pr_model.predict([clear_example1])[0])
print("Predicted cloud percentage for clear_example2:", pr_model.predict([clear_example2])[0])

print("Predicted cloud cover for partly_cloudy_example1:", pr_model.predict([partly_cloudy_example1])[0])
print("Predicted cloud cover for partly_cloudy_example2:", pr_model.predict([partly_cloudy_example2])[0])

print("Predicted cloud cover for cloudy_example1:", pr_model.predict([cloudy_example1])[0])
print("Predicted cloud cover for cloudy_example2:", pr_model.predict([cloudy_example2])[0])

print("Predicted cloud cover for overcast_example1:", pr_model.predict([overcast_example1])[0])
print("Predicted cloud cover for overcast_example2:", pr_model.predict([overcast_example2])[0])