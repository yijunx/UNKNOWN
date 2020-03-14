import os

# os.environ["SPARK_HOME"] = "/usr/local/Cellar/apache-spark/1.5.1/"
# os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3.7"
# os.environ['JAVA_HOME'] = "/usr/libexec/java_home -v 1.8"

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import Row

sc = SparkContext()
sqlContext = SQLContext(sc)


l = [('Ankit', 25), ('Jalfaizy', 22), ('saurabh', 20), ('Bala', 26)]
rdd = sc.parallelize(l)
people = rdd.map(lambda x: Row(name=x[0], age=int(x[1])))
schemaPeople = sqlContext.createDataFrame(people)

print(schemaPeople)

nums = sc.parallelize([1, 2, 3, 4])


list_p = [('John', 19), ('Smith', 29), ('Adam', 35), ('Henry', 50)]
rdd = sc.parallelize(list_p)
rdd.map(lambda x: Row(name=x[0], age=int(x[1])))

# step 1 basic operation with pyspark
url = "https://raw.githubusercontent.com/guru99-edu/R-Programming/master/adult_data.csv"
from pyspark import SparkFiles
sc.addFile(url)
sqlContext = SQLContext(sc)
df = sqlContext.read.csv(SparkFiles.get("adult_data.csv"), header=True, inferSchema=True)
# inferschema = True will infer the datatype, if false, all string..

df.printSchema()
df.show(5, truncate=False)


df.select('age', 'fnlwgt').show(5)

df.groupBy("education").count().sort("count", ascending=True).show()


df.describe().show()
# cannot
