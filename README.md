# 4680-ML-assignment
Findings and Model Evaluation Summary
Dataset taken from Kaggle: https://www.kaggle.com/datasets/nikhil7280/weather-type-classification/data

Decided to apply classification to predict cloud cover (Clear, Partly Cloudy, or Overcast)
Read data from the .csv and used Humidity, Precipitation, and Visibility as my independent variables.
I decided to use the same SVM model that was shown in class.
The predictions were relatively accurate, except that "Cloudy" would never appear as a prediction.
This was perhaps due to the low sample size for "Cloudy" in the dataset, resulting in "Partly Cloudy" appearing instead.
<img width="523" height="207" alt="image" src="https://github.com/user-attachments/assets/09514189-3aa4-454c-80b6-3fc061a83b07" />

The Kaggle dataset is also stated to have statistical outliers:
"This dataset is synthetically produced and does not convey real-world weather data. It includes intentional outliers to provide opportunities for practicing outlier detection and handling. The values, ranges, and distributions may not accurately represent real-world conditions, and the data should primarily be used for educational and experimental purposes." 

I attempted to add "Temperature" and "UV Index" as additional independent variables, but the results stayed largely the same without "Cloudy" appearing for predictions.

In order to apply Regression to the same target to predict, I had to assign arbitrary ranges to the four cloud cover results, as I would instead be predicting cloudiness percentage. I decided to try using a Polynomial Regression model. The predictions for "Clear" and "Partly Cloudy" appeared accurate, but the prediction for "Cloudy" was still in the range for "Partly Cloudy". Likewise, the prediction for "Overcast" ended up in the range for "Cloudy". 
<img width="554" height="207" alt="image" src="https://github.com/user-attachments/assets/6a35add5-0bee-4aab-89b2-4fbf3b15e164" />

I don't believe that Regression is a good fit for my target to predict, since I had to assign arbitrary values to the categories. Although classification had the problem with "Cloudy" not appearing as a prediction, I believe this was likely due to low sample size within the dataset, and the prediction was accurate for all the other cloud cover categories. Perhaps I should have hand-picked an equal sample size from each of the cloud cover categories rather than extracted the entire dataset. 
