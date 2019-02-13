# $ (base) C:\Users\Rubens\Documents>aws configure add-model --service-model file://forecastquery-2018-06-26.normal.json --service-name forecastquery

# $ (base) C:\Users\Rubens\Documents>aws configure add-model --service-model file://forecast-2018-06-26.normal.json --service-name forecast

import boto3
from time import sleep
import subprocess
import matplotlib.pyplot as plt
%matplotlib inline

#! pip install awscli --upgrade
#! pip install boto3 --upgrade
#! pip install six==1.11.0
#! pip install botocore==1.12.92
#! pip install s3transfer==0.2.0

session = boto3.Session(region_name='us-west-2') #us-east-1 is also supported

forecast = session.client(service_name='forecast')
forecastquery = session.client(service_name='forecastquery')

forecast.list_recipes()

import pandas as pd
trf = pd.read_csv("weather.csv", dtype = object)
trf['itemname'] = 'weather_12'
trf.head(3)

s3 = session.client('s3')
accountId = boto3.client('sts').get_caller_identity().get('Account')

# $ (base) C:\Users\Rubens\Anaconda3>python setup_forecast_permissions.py bucket-2

roleArn = 'arn:aws:iam::%s:role/amazonforecast'%accountId
DATASET_FREQUENCY = "H" 
TIMESTAMP_FORMAT = "yyyy-MM-dd hh:mm:ss"

project = 'test_demo'
datasetName= project+'_dataset'
datasetGroupName= project +'_group'
s3DataPath = "s3://"+"bucket-2"+"/"+"weather.csv"
s3DataPath

schema ={
   "Attributes":[    
      { "AttributeName":"timestamp",     "AttributeType":"timestamp"    },      
      { "AttributeName":"season",        "AttributeType":"integer"      },      
      { "AttributeName":"holiday",       "AttributeType":"integer"      },      
      { "AttributeName":"workday",       "AttributeType":"integer"      },      
      { "AttributeName":"weather",       "AttributeType":"integer"      },     
      { "AttributeName":"temperature",   "AttributeType":"float"        },       
      { "AttributeName":"atemp",         "AttributeType":"float"        }, 
      { "AttributeName":"humidity",      "AttributeType":"integer"      },       
      { "AttributeName":"windspeed",     "AttributeType":"float"      },       
      { "AttributeName":"casual",        "AttributeType":"integer"      },      
      { "AttributeName":"registered",    "AttributeType":"integer"      },         
      { "AttributeName":"demand",         "AttributeType":"float"       },      
      { "AttributeName":"item_id",       "AttributeType":"string"       }       
  ]
}

response=forecast.create_dataset(
                    Domain="RETAIL",
                    DatasetType='TARGET_TIME_SERIES',
                    DataFormat='CSV',
                    DatasetName=datasetName,
                    DataFrequency=DATASET_FREQUENCY, 
                    TimeStampFormat=TIMESTAMP_FORMAT,
                    Schema = schema
                   )

predictorName= project+'_predname'
forecastHorizon = 24

createPredictorResponse=forecast.create_predictor(RecipeName='forecast_MQRNN',DatasetGroupName= datasetGroupName ,PredictorName=predictorName, 
  ForecastHorizon = forecastHorizon)

predictorVerionId=createPredictorResponse['VersionId']

while True:
    predictorStatus = forecast.describe_predictor(PredictorName=predictorName,VersionId=predictorVerionId)['Status']
    print(predictorStatus)
    if predictorStatus != 'ACTIVE' and predictorStatus != 'FAILED':
        sleep(30)
    else:
        break

CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
ACTIVE

  
forecastquery.get_accuracy_metrics(PredictorName=predictorName)
forecast.deploy_predictor(PredictorName=predictorName)
deployedPredictorsResponse=forecast.list_deployed_predictors()
print(deployedPredictorsResponse)

while True:
    deployedPredictorStatus = forecast.describe_deployed_predictor(PredictorName=predictorName)['Status']
    print(deployedPredictorStatus)
    if deployedPredictorStatus != 'ACTIVE' and deployedPredictorStatus != 'FAILED':
        sleep(30)
    else:
        break
print(deployedPredictorStatus)

CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
ACTIVE
ACTIVE

forecastResponse = forecastquery.get_forecast(
    PredictorName=predictorName,
    Interval="hour",
   Filters={"item_id":"weather_12"}
)
print(forecastResponse)

d = pd.DataFrame.from_dict(forecastResponse['Forecast'])
df = pd.DataFrame.from_dict(d.loc['mean']['Predictions']).dropna().rename(columns = {'Val':'p50mean'})
df.plot()

forecastInfoList= forecast.list_forecasts(PredictorName=predictorName)['ForecastInfoList']
forecastId= forecastInfoList[0]['ForecastId']

outputPath="s3://"+"bucket-2"+"/output"

forecastExportResponse = forecast.create_forecast_export_job(ForecastId=forecastId, OutputPath={"S3Uri": outputPath,"RoleArn":roleArn})

forecastExportJobId = forecastExportResponse['ForecastExportJobId']

while True:
    forecastExportStatus = forecast.describe_forecast_export_job(ForecastExportJobId=forecastExportJobId)['Status']
    print(forecastExportStatus)
    if forecastExportStatus != 'ACTIVE' and forecastExportStatus != 'FAILED':
        sleep(30)
    else:
        break
    
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING
CREATING

s3.list_objects(Bucket="bucket-2",Prefix="output")
