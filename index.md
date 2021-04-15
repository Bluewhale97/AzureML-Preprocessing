## Introduction

Data preprocessing typically involves into the process of data preparation. We sometimes name the data preparation as the fearture engineering and data cleansing. In the assignments of preprocessing, we generally include converting data to a different scale type, converting data to a different scale and data transformation and dimensionality reduction.

In this exercise of data preprocessing in Azure, we are going to glance at a few popularized techniques in data preprocessing.

## 1. Scaling numeric features

Scaling numeric features in this tutorial is more like a process to normalize features so they are on a same scale, preventing to produce 
disproportionate coefficients on models. 

For example:
```python
# Define preprocessing for numeric columns (scale them)
numeric_features = [6,7,8,9]
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])
```

## 2. Encoding

In this tutorial, encoding categorical variables is what we usually discuss, called factorization, it is to convert categorical features into numeric representations. Actually the factoring process includes ordered type and unordered type. We could discuss it in another article.

Let's see this example
```python
# Define preprocessing for categorical features (encode them)
categorical_features = [0,1,2,3,4,5]
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
```

## 3. Modeling

Now we use the methods of preprocessing to train the model, we will combine the preprocessing steps and create preprocessing and training pipeline, then train a linear regression.

```python
# Train the model
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
import numpy as np
# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Create preprocessing and training pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', GradientBoostingRegressor())])


# fit the pipeline to train a linear regression model on the training set
model = pipeline.fit(X_train, (y_train))
print (model)

## 4. Prediction

The prediction is to perform the model to the vaildation data set, we want to know the evaluation and performance of our model now.

```python
# Get predictions
predictions = model.predict(X_test)

# Display metrics
mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
r2 = r2_score(y_test, predictions)
print("R2:", r2)

# Plot predicted vs actual
plt.scatter(y_test, predictions)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('Daily Bike Share Predictions')
z = np.polyfit(y_test, predictions, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test), color='magenta')
plt.show()
```

The performance shows:

![image](https://user-images.githubusercontent.com/71245576/114891239-b5e3c780-9dd9-11eb-82e6-9ca16a9b501d.png)

```
The R2 is 0.79, it seems like performing good.

Now let's try a random forest model:

```python
# Use a different estimator in the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', RandomForestRegressor())])


# fit the pipeline to train a linear regression model on the training set
model = pipeline.fit(X_train, (y_train))
print (model, "\n")

# Get predictions
predictions = model.predict(X_test)

# Display metrics
mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
r2 = r2_score(y_test, predictions)
print("R2:", r2)

# Plot predicted vs actual
plt.scatter(y_test, predictions)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('Daily Bike Share Predictions - Preprocessed')
z = np.polyfit(y_test, predictions, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test), color='magenta')
plt.show()
```

There is not improved significantly at all, the R2 is 0.79 as well.

![image](https://user-images.githubusercontent.com/71245576/114891941-533efb80-9dda-11eb-9529-ae5433a76b10.png)

## 4. Scoring and inferencing

This part is simple, after we trained a model, we can save it for predicting labels for new data in future, it is often called scoring and inferencing.

```python
import joblib

# Save the model as a pickle file
filename = './models/bike-share.pkl'
joblib.dump(model, filename)
```

Now, we load it for new data:

```python
# Load the model from the file
loaded_model = joblib.load(filename)

# Create a numpy array containing a new observation (for example tomorrow's seasonal and weather forecast information)
X_new = np.array([[1,1,0,3,1,1,0.226957,0.22927,0.436957,0.1869]]).astype('float64')
print ('New sample: {}'.format(list(X_new[0])))

# Use the model to predict tomorrow's rentals
result = loaded_model.predict(X_new)
print('Prediction: {:.0f} rentals'.format(np.round(result[0])))
```

The prediction is 109 rentals, for the new sample:

![image](https://user-images.githubusercontent.com/71245576/114892341-ae70ee00-9dda-11eb-8f6d-3c280370147c.png)

Let's try a 5-day rental predictions:

```python
# An array of features based on five-day weather forecast
X_new = np.array([[0,1,1,0,0,1,0.344167,0.363625,0.805833,0.160446],
                  [0,1,0,1,0,1,0.363478,0.353739,0.696087,0.248539],
                  [0,1,0,2,0,1,0.196364,0.189405,0.437273,0.248309],
                  [0,1,0,3,0,1,0.2,0.212122,0.590435,0.160296],
                  [0,1,0,4,0,1,0.226957,0.22927,0.436957,0.1869]])

# Use the model to predict rentals
results = loaded_model.predict(X_new)
print('5-day rental predictions:')
for prediction in results:
    print(np.round(prediction))
```
The result:

![image](https://user-images.githubusercontent.com/71245576/114892478-cf394380-9dda-11eb-9c94-98982cd63df4.png)

Reference:

Train and evaluate regression models, retrieved from https://docs.microsoft.com/en-us/learn/modules/train-evaluate-regression-models/
