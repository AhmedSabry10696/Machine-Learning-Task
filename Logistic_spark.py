
"""
Important:
    
To download and setup pyspark in anaconda:

In anaconda propmt you can write:
    conda install -c conda-forge pyspark 
    or
    conda install -c conda-forge/label/cf201901 pyspark 

"""


"""--------------------------------------------------------------------------
Load Data
-------------------------------------------------------------------------"""

#Load the CSV file into a RDD

from pyspark import SparkContext,SparkConf
conf=SparkConf().setAppName("Dicision_Tree").setMaster("local[2]")
SpContext=SparkContext.getOrCreate()
irisData = SpContext.textFile("Bank-Loan.csv")
#spcontext is to read and load a csv file you want to use into an RDD
irisData.cache()
#.cache is for caching the dataset you want to use for speeding up processing.
#irisData.count()
#Remove the first line (contains headers)
dataLines = irisData.filter(lambda x: "home_ownership" not in x)
#filter is used to remove the line (element) from the dataset based on a condition.
#lambda x is for input arguemment for the filter function.


"""--------------------------------------------------------------------------
Cleanup Data
by converting the data into a dense vector: by changing non numeric values into numeric.
-------------------------------------------------------------------------"""

from pyspark.sql import Row
#Create a Data Frame from the data
parts = dataLines.map(lambda l: l.split(","))
#The job of map is to clean the data, filtering, splitting or converting the data

irisMap = parts.map(lambda p: Row(home_ownership=str(p[0]),\
                                annual_inc=float(p[1]), \
                                loan_amount=float(p[2]), \
                                SPECIES=p[3] ))
#float is converting all elements into float putting each one in attribute.
# and for elemenating any column you want to remove.
# Infer the schema, and register the DataFrame as a table.
from pyspark.sql import SparkSession
SpSession=SparkSession(SpContext)#creating a spark session.
irisDf = SpSession.createDataFrame(irisMap)
irisDf.cache()
#Add a numeric indexer for the label/target column
from pyspark.ml.feature import StringIndexer#it takes string values converting them to numericvalues

stringIndexer = StringIndexer(inputCol="SPECIES", outputCol="IND_SPECIES")
#To convert the string values into numeric values.
si_model = stringIndexer.fit(irisDf)
irisNormDf = si_model.transform(irisDf)

stringIndexer2 = StringIndexer(inputCol="home_ownership", outputCol="IND_home_ownership")
si_model2 = stringIndexer2.fit(irisNormDf)
irisNormDf = si_model2.transform(irisNormDf)

irisNormDf.select("SPECIES","IND_SPECIES").distinct().show()
irisNormDf.select("home_ownership","IND_home_ownership").distinct().show()

irisNormDf.cache()



"""--------------------------------------------------------------------------
Perform Data Analytics
-------------------------------------------------------------------------"""

#See standard parameters
irisNormDf.describe().show()



"""--------------------------------------------------------------------------
Prepare data for ML
-------------------------------------------------------------------------"""

#Transform to a Data Frame for input to Machine Learing
#Drop columns that are not required (low correlation)

from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
SpSession=SparkSession(SpContext)
#Here, we are creating a spark session from a spark context.
def transformToLabeledPoint(row) :
    lp = ( row["SPECIES"], row["IND_SPECIES"], \
                Vectors.dense([row["IND_home_ownership"],\
                        row["annual_inc"], \
                        row["loan_amount"]]))
    return lp
#In order to convert the loaded dataset nto a labeled point.
irisLp = irisNormDf.rdd.map(transformToLabeledPoint)
irisLpDf = SpSession.createDataFrame(irisLp,["species","label", "features"])
irisLpDf.select("species","label","features").show(10)
irisLpDf.cache()
#This is for creating a data frame to apply machine learning upon it.

"""--------------------------------------------------------------------------
Perform Machine Learning
-------------------------------------------------------------------------"""
#Split into training and testing data
(trainingData, testData) = irisLpDf.randomSplit([0.9, 0.1])
#Here, we split the dataset for a test and a train set.
trainingData.count()
#Here, we count then show the data. 
testData.count()
testData.show()
print("welcomeeeeeeeeee")
print ("we",trainingData.show())
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
#Here, we are importing the classifier and applying it.
#Create the model
dtClassifer = LogisticRegression( labelCol="label",\
                featuresCol="features")
dtModel = dtClassifer.fit(trainingData)


#Predict on the test data
predictions = dtModel.transform(testData)
predictions.select("prediction","species","label").show()

#Evaluate accuracy
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", \
                    labelCol="label",metricName="accuracy")
acc = evaluator.evaluate(predictions)      

#Draw a confusion matrix
predictions.groupBy("label","prediction").count().show()

