# Analysis of New York traffic data using Azure Cloud, PySpark and Databricks
Assignment as part of Big Data Engineering subject

## Overview
We have been provided the task of analyzing the monthly New York City (NYC) taxi data of the yellow and green taxis from the period of Jan 2019 up to July 2021. 
This report is aimed at helping taxi drivers make informed decisions about potential ways to improve their earnings potential. Other aspects of this report may help policy makers and/or regulatory bodies make informed decisions about changes to taxi operations within NYC.
In addition, two machine learning algorithms have been implemented to gain insight into the factors that influence of taxi drivers’ earnings potential. 
All analysis was performed using a Databricks notebook utilizing various libraries for data cleaning, querying and machine learning processes.

## Data cleaning

There were 124,048,218 rows in the yellow taxi dataset and 8,348,567 in the green dataset for a total of 132,396,785 combined rows. 
A data dictionary is also available on the NYC taxi website which explains various aspects of the data which I won’t go into detail here. It was noticed that ehail_fee and congestion_surcharge exists in the dataset but not in either data dictionary. Additionally, trip_type is unique to the green taxi dataset. 
The names of the pickup and drop-off timestamps were slightly different and renamed to either pickup_timestamp or dropoff_timestamp respectively.
The trip distance was also in miles, so I altered the name to include miles in the columns name for both datasets.
As the yellow dataset lacked ehail_fee and trip type column, additional columns were added with values set at 0, as there shouldn’t be an ehail fee and the trip type value of 0 can be categorized as ‘unknown’. I added a final column called taxi_type to distinguish the origin of the row across the dataset.
The column order of the green dataset was rearranged to match the yellow dataset. The schema of both datasets was set according to Appendix Table 1. The datasets were subsequently merged.

## Analysis

### 1. Which day of the week had the most number of trips 

```
SELECT day_of_week,
       CASE day_of_week
         WHEN 1 THEN 'Sunday'
         WHEN 2 THEN 'Monday'
         WHEN 3 THEN 'Tuesday'
         WHEN 4 THEN 'Wednesday'
         WHEN 5 THEN 'Thursday'
         WHEN 6 THEN 'Friday'
         WHEN 7 THEN 'Saturday'
       end,
       Count(*) AS Trips_per_day
FROM   taxi_df
GROUP  BY day_of_week
ORDER  BY trips_per_day DESC
LIMIT  1 
```

![image](https://user-images.githubusercontent.com/53500810/206883309-23a2296b-8560-4327-b2ef-054bc8638889.png)


2. Display the number of trips for each hour (12am - 1am is hour 0,  1am - 2am is hour 1 etc.)

```
SELECT 24_hour,
       Count(24_hour) AS Num_trips
FROM   taxi_df
GROUP  BY 24_hour
ORDER  BY 24_hour ASC 
```

![image](https://user-images.githubusercontent.com/53500810/206883336-6371630e-0931-46ad-ba39-f6346ad34f0c.png)

3. What was the average, median, minimum and maximum trip duration in seconds?

```
SELECT taxi_type,
       Percentile_approx(time_of_trip_seconds, 0.5) AS median_trip_time_seconds,
       Min(time_of_trip_seconds)                    AS min_trip_time_seconds,
       Max(time_of_trip_seconds)                    AS max_trip_time_seconds,
       Round(Avg(time_of_trip_seconds), 2)          AS avg_trip_time_seconds
FROM   taxi_df
GROUP  BY taxi_type 
```

![image](https://user-images.githubusercontent.com/53500810/206883370-4247484f-2c3a-4af3-a946-f8491a64e4d3.png)


## Machine Learning

Two models were implemented to predict the total fare (without using the Fare Amount) using 2019 and 2020 data to preduct 2021 data.

### list of categorial variables and numeric variables

```
## Categorical columns
cat_cols = ['VendorID', 'RatecodeID','PU_Borough','DO_Borough', 'payment_type', 'taxi_type', 'day_of_week'] 
## Numerical columns
num_cols = ['passenger_count', 'extra','mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge', 'congestion_surcharge', 'trip_distance_km', 'year','month', '24_hour', 'time_of_trip_seconds', 'speed_km_h']
     
```

### Encode taxi type (yellow = 0, green = 1)
```
# encode value
df_taxi_merged_filtered = df_taxi_merged_filtered.withColumn("taxi_type",when(df_taxi_merged_filtered.taxi_type == "yellow","0").
                         when(df_taxi_merged_filtered.taxi_type == "green","1"))

# fix schema now that column is integer
df_taxi_merged_filtered = df_taxi_merged_filtered.withColumn("taxi_type", F.col("taxi_type").astype(IntegerType()))
```

### Train/Test split
```
## split into test and train

# train data is all except 2021
train_data = df_taxi_merged_filtered[df_taxi_merged_filtered.year <=2020]
# smaller dataset
train_data_small = train_data.sample(False, 0.001, 42)

# test data is only 2021
test_data = df_taxi_merged_filtered[df_taxi_merged_filtered.year >=2021]
```

## Use best predictors of Random Forest
```
# function for getting relevant info 
def ExtractFeatureImp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))


# Only get top 10 values, whole tale saved in report
ExtractFeatureImp(basic_rf_model.stages[-1].featureImportances, df2, "features").head(10)


# categorical columns
cat_cols = ['VendorID', 'RatecodeID','PULocationID','DOLocationID', 'payment_type', 'taxi_type', 'day_of_week'] 
## Numerical columns
num_cols = ['passenger_count', 'extra','mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge', 'congestion_surcharge', 'trip_distance_km', 'year','month', '24_hour', 'time_of_trip_seconds', 'speed_km_h']
     
```

### Create a Pipeline
```
# adding columns from dataframe
assembler = VectorAssembler(inputCols=cat_cols+num_cols, outputCol="features")

# define basic model
rf = RandomForestRegressor(labelCol="label", featuresCol="features",seed = 42, numTrees=10, subsamplingRate = 0.7)

# create pipeline
pipeline = Pipeline(stages=[assembler, rf])

# Fit model
basic_rf_model = pipeline.fit(train_data_small)

# Transformed model
df2 = basic_rf_model.transform(train_data_small)
```

## Hyperparameter tuning
```
# get most important features
top_num = ['trip_distance_km', 'time_of_trip_seconds', 'tip_amount', 'tolls_amount', 'speed_km_h', 'extra', 'mta_tax']
top_cat = ['RatecodeID', 'DOLocationID', 'payment_type']

# create assembler
assembler = VectorAssembler(inputCols=top_cat+top_num, outputCol="features")

# create pipeline
pipeline = Pipeline(stages=[assembler, rf])

# create grid
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 15, 20]) \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .build()

# implement TrainValidationSplit
tvs = TrainValidationSplit(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(), # default rmse
                          trainRatio=0.8)

# fit model
tvs_rf = tvs.fit(train_data_small) #47.02 minutes
```

### Get best results
```
best_Model = tvs_rf.bestModel

print(' Best number of trees', best_Model.stages[-1].getNumTrees) #15
print(' Best max depth ', best_Model.stages[-1].getNumTrees) #15
```

## Use best hyperparamers
```
# create assembler
assembler = VectorAssembler(inputCols=top_cat+top_num, outputCol="features")

# define model with hyperparameters
rf_total_df = RandomForestRegressor(labelCol="label", featuresCol="features", maxDepth = 15,numTrees =15)

# create pipeline
pipeline = Pipeline(stages=[assembler, rf_total_df])

# train model on restricted dataset 10%
best_rf_model = pipeline.fit(train_data.sample(False, 0.1, 42)) 
```

## Get predictions are evaluation
```
# get predictions and add to dataframe
predictions = best_rf_model.transform(train_data)

# evaluate model
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")

# Find rmse
rmse = evaluator.evaluate(predictions)

print('rmse', rmse)
```
rmse 151.65738123425095




## Test on 2021 data
```
# get predictions and add to dataframe
test_predictions = best_rf_model.transform(test_data)
     

# evaluate model
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")

# Find rmse
rmse = evaluator.evaluate(test_predictions)
print('rmse', rmse)
     
rmse 111.2976170401833

# compare predictions to actual values
test_predictions_clear = test_predictions.withColumn('prediction',F.round(test_predictions["prediction"],2))
test_predictions_clear.select(['label', 'prediction']).show(10)
```

## Plotting 5% of test data for visuals
![image](https://user-images.githubusercontent.com/53500810/206883670-5acee48c-d1f2-42ae-a407-40108ce1b335.png)


