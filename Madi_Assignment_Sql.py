

                                        ########## SQL ###########
import re
import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import functions as sf
from pyspark.sql import Row

spark = SparkSession \
    .builder \
    .appName("Madi_Assignment") \
    .getOrCreate()

# DataFrame creation from JSON file
purchases = os.path.join('data', 'purchases.json')
df = spark.read.json(purchases)

spark = SparkSession \
    .builder \
    .appName("Madi_Assignment_Sql") \
    .getOrCreate()
# DataFrame creation from JSON file
df = spark.read.json("/Users/madisubaiti/PycharmProjects/Spark2/x/data/purchases.json")
# Register as a table
df.createOrReplaceTempView("purchases")
df.registerTempTable("purchases")


    ##########1. top ten most purchased products
df1 = spark.sql(
    "SELECT product_id, COUNT(*) AS count" +
    " FROM purchases" +
    " GROUP BY product_id" +
    " ORDER BY count desc" +
    " LIMIT 10")
print 'The 10 most purchased products are:'
df1.show(10)

    ##############2. Purchase percentage of each product type (item_type)
dff2 = spark.sql(
  "SELECT item_type, COUNT(*) as count," 
  " COUNT(item_type) * 100 / (SELECT COUNT(*) FROM purchases) as percent" 
  " FROM purchases" 
  " GROUP BY item_type")

print 'Each product percentage is:'
dff2.show()

    ###########3. Shop that has sold more products

df3 = spark.sql(
    "SELECT shop_id, COUNT(*) AS count" +
    " FROM purchases" +
    " GROUP BY shop_id" +
    " ORDER BY count desc" +
    " LIMIT 1")
print 'Most products sold were from:'
df3.show()

    ###########4. Shop that has billed more money
df4 = spark.sql(
    "SELECT shop_id,SUM(price)total_sales"
    " FROM purchases"
    " GROUP BY shop_id"
    " ORDER BY total_sales "
    " LIMIT 1"
)

print'Most billed from:'
df4.show()

    ######### 5 Divide world into 5 geographical areas based on longitude (location.lon) and add a column
    ## with geographical area name, for example “area1”, “area2”, ... (longitude values are not important).

locarea = spark.sql(
    " SELECT *,"
    " CASE WHEN location.lon <= -109.8768 THEN 'area #1'"
    " WHEN location.lon <= -32.9207 THEN 'area #2'"
    " WHEN location.lon <= 36.9976 THEN 'area #3'"
    " WHEN location.lon <= 103.2355 THEN 'area #4'"
    " ELSE 'area #5'"
    " END AS locarea"
    " from purchases"
)

locarea.createOrReplaceTempView("t_locarea")

print'World divided by longitude'

locarea.show()

    ######### a. In which area is PayPal most used

paypal2 = spark.sql(
    " SELECT payment_type, locarea, COUNT(locarea) as COUNT"
    " FROM t_locarea "
    " WHERE payment_type = 'paypal'"
    " GROUP BY locarea, payment_type"
    " ORDER BY count DESC"
    " LIMIT 1"
)

print'PayPal most used in:'
paypal2.show()



    ######### b. Top 3 most purchased products in each area

purch3 = spark.sql (
    " SELECT * FROM "
    " (SELECT product_id,locarea,COUNT(product_id) as count_prod"
    " FROM t_locarea WHERE locarea = 'area #1'"
    " GROUP BY locarea, product_id"
    " ORDER BY count_prod DESC"
    " LIMIT 3)"
    " UNION ALL"
    " (SELECT product_id, locarea, COUNT(product_id) as count_prod"
    " FROM t_locarea WHERE locarea = 'area #2'"
    " GROUP BY locarea, product_id"
    " ORDER BY count_prod DESC"
    " LIMIT 3)"
    " UNION ALL"
    " (SELECT product_id, locarea, COUNT(product_id) as count_prod"
    " FROM t_locarea WHERE locarea = 'area #3'"
    " GROUP BY locarea, product_id"
    " ORDER BY count_prod DESC"
    " LIMIT 3)"
    " UNION ALL"
    " (SELECT product_id,locarea, COUNT(product_id) as count_prod"
    " FROM t_locarea WHERE locarea = 'area #4'"
    " GROUP BY locarea, product_id"
    " ORDER BY count_prod DESC"
    " LIMIT 3)"
    " UNION ALL"
    " (SELECT product_id, locarea, COUNT(product_id) as count_prod"
    " FROM t_locarea WHERE locarea = 'area #5'"
    " GROUP BY locarea, product_id"
    " ORDER BY count_prod DESC"
    " LIMIT 3)"
)

print'Each areas top 3 purchased products:'
purch3.show()


    ########## c. Area that has billed less money

locareabill = spark.sql (
    " SELECT locarea, SUM(price) x"
    " FROM t_locarea"
    " GROUP BY locarea"
    " ORDER BY x "
    " LIMIT 1"
)

print'This area billed the least:'
locareabill.show()


    ############ 6- Products that do not have enough stock for purchases made

df_s1 = spark.read.csv("/Users/madisubaiti/PycharmProjects/Spark2/x/data/stock.csv", header=True)
df_s1.createOrReplaceTempView("stock")
df_s1.registerTempTable("stock")


stock1 = spark.sql(
    " SELECT a.product_id, b.quantity, count(b.product_id) as purchases"
    " FROM purchases a join stock b on "
    "  a.product_id = b.product_id "
    " GROUP BY a.product_id,b.quantity"
    " HAVING a.product_id > b.quantity"

)

print'These Products do not have enough stock for purchases made'
stock1.show()

