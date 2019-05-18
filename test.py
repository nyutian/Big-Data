#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr,log
from pyspark.ml.recommendation import ALSModel
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
import pyspark.sql.functions as F
import numpy as np

def main(spark, model_file, data_file,count):
    df = spark.read.parquet(data_file).repartition(2000,['userIndex'])
    if count =='log':
        df = df.withColumn("count",log(col("count")+1))
    elif count =='drop1':
        df = df.filter('count>1')
    elif count =='drop2':
        df = df.filter('count>2')
    model = ALSModel.load(model_file)
    test_user = df.select('userIndex').distinct()
    predictions = model.transform(df)
    actual = predictions.groupBy("userIndex").agg(expr("collect_set(trackIndex) as tracks"))
    rec = model.recommendForUserSubset(test_user,500)
    a= rec.select('userIndex','recommendations.trackIndex')
    b=a.join(actual,['userIndex']).select('trackIndex','tracks').rdd
    metrics = RankingMetrics(b)
    result = metrics.meanAveragePrecision
    print(result)
    np.savetxt('drop2.txt',np.array([result]))
    pass

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('test').getOrCreate()
    model_file = sys.argv[1]
    data_file = sys.argv[2]
    count = sys.argv[3]
    # Call our main routine
    main(spark, model_file, data_file, count)
