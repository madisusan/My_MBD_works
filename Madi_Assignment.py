import re
import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import functions as sf
from pyspark.sql import Row
from pyspark.sql.window import Window

spark = SparkSession \
    .builder \
    .appName("Madi_Assignment") \
    .getOrCreate()

# DataFrame creation from JSON file
purchases = os.path.join('data', 'purchases.json')
df = spark.read.json(purchases)

                                        ######## API #########
    #1. top ten most purchased products
df.printSchema()
df.describe().show
df.show()

products=df.groupBy(df.product_id).agg(sf.count("*").alias("x")).orderBy("x", ascending=False)
products.show(10)

    #2. Purchase percentage of each product type (item_type)
prod_per=df.groupBy(df.item_type).agg((sf.count("*")/df.count()).alias("y")).show()
prod_per.show()

    #3. Shop that has sold more products
shop1=df.groupBy(df.shop_id).agg(sf.count(df.product_id).alias("z")).orderBy("z", ascending=False)
shop1.show(2)

    #4. Shop that has billed more money
price1=df.groupBy(df.shop_id).agg(sf.sum(df.price).alias("w")).orderBy("w", ascending=False)
price1.show(2)

    #5. Divide world into 5 geographical areas based in longitude (location.lon) and add a column
#with geographical area name, for example “area1”, “area2”, ... (longitude values are not important).

#long1=df.agg(sf.max(df.location.lon).alias("m")).orderBy("m", ascending=True)
#long1.show()
#long1=df.agg(sf.min(df.location.lon).alias("m")).orderBy("m", ascending=True)
#long1.show()
#df.select(df.location.lon).show(40)


d5 = df.withColumn( 'area',sf.when((sf.col("location.lon") <= -109.8768),'area #1')
.when((sf.col("location.lon") <= -32.9207), 'area #2')
.when((sf.col("location.lon") <= 36.9976), 'area #3')
.when((sf.col("location.lon") <= 103.2355), 'area #4').otherwise('area #5'))
d5.show()


    # a. In which area is PayPal most used

paypal1 = d5.where(d5.payment_type == "paypal")\
    .groupBy(d5.area)\
    .agg(sf.count(d5.product_id).alias("paypal_count"))\
    .orderBy("paypal_count", ascending=False)
paypal1.show(1)


    # b. Top 3 most purchased products in each area
    #  partitionBy and withColumn functions

 print "The top products purchased per area:"
prod_x_area= d5.groupBy("area","product_id")\
    .agg(sf.count("*").alias("counts"))\
    .orderBy("area","counts",ascending=False)

prod_x_area.withColumn("row_num", sf.row_number().over(Window.partitionBy("area").orderBy(prod_x_area['counts'].desc())))\
    .filter(sf.col('row_num')<=3).show(20)


# #c. Area that has billed less money

arealow = d5.groupBy(d5.area).agg(sf.sum(d5.price).alias("least bill")).orderBy("least bill", ascending=True)
arealow.show(1)



    #6 Products that do not have enough stock for purchases made

df2 = spark.read.option("header", "true") \
               .option("inferSchema", "true") \
               .csv("data/stock.csv")
df2 = spark.read.csv("/Users/madisubaiti/PycharmProjects/Spark2/x/data/stock.csv", header=True)
df2.show()


group = df.groupBy(df.product_id).agg(sf.count(df.product_id).alias("prod_count"))
prod1= group.join(df2, on = "product_id")
df.where("prod_count" > df2.quantity).show()


