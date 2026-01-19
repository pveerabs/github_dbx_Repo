# Databricks notebook source
# DBTITLE 1,Import the libraries
 
from pyspark.sql.functions import col,count,when
from pyspark.sql.types import StructType,StructField, StringType, IntegerType,FloatType
from pyspark.sql.functions import max,min,avg,sum,count,round
from pyspark.sql.functions import lit
from pyspark.sql.functions import when
from pyspark.sql.functions import expr
from pyspark.sql.functions import variance,stddev,skewness,kurtosis
import matplotlib.pyplot as plt 




# COMMAND ----------

# DBTITLE 1,Read the data from the file and create

df_emp = spark.table("employee")
display(df_emp)

# COMMAND ----------

# DBTITLE 1,Count the records in the DataFrame

print(df_emp.count())

# COMMAND ----------

# DBTITLE 1,check the dataframe datatypes

df_emp.printSchema()

# COMMAND ----------

# DBTITLE 1,Check the null values in the dataframe

df_emp_ColNull = df_emp.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in df_emp.columns
])

display(df_emp_ColNull)


# COMMAND ----------

# DBTITLE 1,drop all rows where all columns are null

df_no_all_nulls = df_emp.dropna(how="all")
display(df_no_all_nulls)

# COMMAND ----------

# DBTITLE 1,Copy the dataframe to new dataframe

df_new = df_no_all_nulls
display(df_new)

# COMMAND ----------

# DBTITLE 1,Rename the column names
for col_name in df_new.columns:
    df_new = df_new.withColumnRenamed(col_name, col_name.replace(" ", "_"))

display(df_new)


# COMMAND ----------

# DBTITLE 1,Aggregations
from pyspark.sql.functions import expr
df_aggr = df_new.selectExpr(
    "count(*) as `Total Records`",
    "sum(Monthly_Income) as `Total Income`",
    "round(avg(Monthly_Income),2) as `Average Income`",
    "min(Monthly_Income) as `Minimum Income`",
    "max(Monthly_Income) as `Maximum Income`"
).show()

# COMMAND ----------

# DBTITLE 1,Fill the null values in the Gender and Marital status columns
df_clean = df_new.fillna({
  "Gender" : "Unknow",
  "Marital_Status" : "Unknow"
})
display(df_clean)

# COMMAND ----------

# MAGIC %md
# MAGIC **Demographic Based Grouping**

# COMMAND ----------

# DBTITLE 1,Group By Gender
df_new.groupBy("Gender").count().show()

# COMMAND ----------

# DBTITLE 1,Marital Status Grouping

df_new.groupBy("Marital_Status").count().orderBy("count", ascending=False).show()

# COMMAND ----------

# DBTITLE 1,Gender Grouping
df_new.groupBy("Gender").count().orderBy("count",ascending=False).show()

# COMMAND ----------

# DBTITLE 1,Group By Age
df_new.groupBy("Age").count().orderBy("Count",ascending=False).show()

# COMMAND ----------

# DBTITLE 1,Group By Gender & Marital_Status
df_new.groupBy("Gender","Marital_Status").count().orderBy("count",ascending=False).show()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC **Department - Level Grouping**

# COMMAND ----------

# DBTITLE 1,Group By Department
df_new.groupBy("Department_Name").count().orderBy("count", ascending=False).show()

# COMMAND ----------

# DBTITLE 1,Group by Department  Vs Performance Rating
df_new.groupBy("Department_name","Performance_Rating").count().orderBy("count",decending=True).show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Salary & Compensation Analysis**

# COMMAND ----------

# DBTITLE 1,Average Monthly Income by Department
df_new.groupBy("Department_name") \
. agg (avg("Monthly_income").alias("Average_Monthly_Income")).show()

# COMMAND ----------

# DBTITLE 1,Salary Hike by Performance Rating
df_new.orderBy("Performance_Rating") \
. agg(avg("Percent_Salary_Hike").alias("Average_hike")).show()

# COMMAND ----------

# DBTITLE 1,Bonus By Department
df_new.groupBy("Department_Name") \
.agg(round(avg("Bonus"),2).alias("Avg_Bonus")).show()  

# COMMAND ----------

# MAGIC %md
# MAGIC **Experience Based Grouping**

# COMMAND ----------

# DBTITLE 1,Years of Experience
df_new.groupBy("Department_Name","Years_of_experience").count().show()

# COMMAND ----------

# DBTITLE 1,Department wise Education
df_new.groupBy("Department_Name","Education").count().show()

# COMMAND ----------

# DBTITLE 1,Education  count
df_new.groupBy("Education").count().show()

# COMMAND ----------

# DBTITLE 1,Perfomance Rating
df_new.groupBy("Performance_Rating").count().show()

# COMMAND ----------

# DBTITLE 1,Performance By Department
df_new.groupBy("Performance_Rating","Department_Name").count().orderBy("count",ascending=False).show()

# COMMAND ----------

# DBTITLE 1,Advanced Grouping
df_new.groupBy(
  "Department_Name",
  "Gender",
  "Performance_Rating"
).agg(
  round(avg("Monthly_Income"),2).alias("avg Income"),
  round(avg("Bonus"),2).alias("avg Bonus")).show()

# COMMAND ----------

# DBTITLE 1,Drop the unwanted columns
df_new = df_new.drop("Phone_Number")
display(df_new)


# COMMAND ----------

# DBTITLE 1,Derived New Columns for the dataframe
from pyspark.sql.functions import when

df_derived_Cols = df_new.withColumn(
    "Income Category",
    when(df_new["Monthly_Income"] > 90000, "High")
    .when(df_new["Monthly_Income"] > 50000, "Medium")
    .otherwise("Low")
)
display(df_derived_Cols)

# COMMAND ----------

# MAGIC %md
# MAGIC **Exploratory Data Analysis (EDA) **

# COMMAND ----------

# DBTITLE 1,First Moment Business Decision
df_new.select(round(avg("Monthly_Income"),2).alias("mean")).show()

# COMMAND ----------

# DBTITLE 1,Second moment business decisions
df_new.select(
    variance("Monthly_Income").alias("variance"),
    stddev("Monthly_Income").alias("std_dev")
).show()

# COMMAND ----------

# DBTITLE 1,Third Moment Business Decisons
df_new.select(
  skewness("Monthly_Income").alias("skewness")
).show()

# > 0 Right skewed ,< 0 left skewed,=0 symmentric

# COMMAND ----------

# DBTITLE 1,Kurtosis
df_new.select(
  kurtosis("Monthly_Income").alias("kurtosis")
).show()
 # >0 Heavly tails(peaked) ,< 0 low tails(peaked),=0 Normal Like

# COMMAND ----------

# MAGIC %md
# MAGIC **IQR (Interquartile Range)-Outlier Detection**

# COMMAND ----------

# DBTITLE 1,IQR
Q1, Q3 = df_new.approxQuantile(
    "Monthly_Income",
    [0.25, 0.75],
    0.01
)

IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)


# COMMAND ----------

# DBTITLE 1,Box Plot
pdf = df_new.select("Monthly_Income").dropna().toPandas()

plt.boxplot(pdf["Monthly_Income"])
plt.title("Box plot of Monthly Income")
plt.ylabel("Monthly_Income")
plt.show()

# COMMAND ----------

# DBTITLE 1,Group Wise Box Plot
pdf = df_new.select("Monthly_Income","Gender").dropna().toPandas()

pdf.boxplot(column = "Monthly_Income" ,by = "Gender")
plt.title("Box plot of Monthly Income by Gender")
plt.suptitle("")
plt.show()

# COMMAND ----------

# DBTITLE 1,Distribution Analysis
plt.hist(pdf["Monthly_Income"],bins = 15)
plt.title("Histogram of Monthly_Income")
plt.xlabel("Monthly Income")
plt.ylabel("Frequncy")
plt.show()



# COMMAND ----------

display(df_new)
