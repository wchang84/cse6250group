from pyspark.sql import SparkSession
from pyspark.sql.types import *
spark = SparkSession.builder.appName("cse6250group").getOrCreate()

# Importing the data from preprocessing
df = spark.read.csv('model data.csv', header=True,schema=StructType([
            StructField('recording', LongType()),
            StructField('Source', StringType()),
            StructField('age', LongType()),
            StructField('sex', StringType()),
            StructField('diagnosis', StringType()),
            StructField('other problems', StringType()),
            StructField('epoches', LongType()),
            StructField('w%', DoubleType()),
            StructField('n1%', DoubleType()),
            StructField('n2%', DoubleType()),
            StructField('n3%', DoubleType()),
            StructField('rem%', DoubleType())]))

df.show()

from pyspark.sql.functions import col,when,lower

df = df.withColumn('diagnosis', when(df.diagnosis == "F",None).otherwise(df.diagnosis))

df.select("diagnosis").distinct().show()

#1hot encoding for gender
df = df.withColumn('is_male', lower(df.sex) == "m")
df = df.withColumn('is_female', lower(df.sex) == "f")
#df.head()
df.head()

df = df.dropna(subset=['w%','n1%', 'n2%', 'n3%', 'rem%', 'age', 'sex', 'diagnosis'])

df.select("diagnosis").distinct().show()

from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=['w%','n1%', 'n2%', 'n3%', 'rem%', 'is_male','is_female'],
    outputCol='features')

transformed_df = assembler.transform(df)


from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator
# Trains a bisecting k-means model.
bkm = BisectingKMeans().setK(11).setSeed(42)
model = bkm.fit(transformed_df)

# Make predictions
predictions = model.transform(transformed_df)

# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

# Shows the result.
print("Cluster Centers: ")
centers = model.clusterCenters()
for center in center