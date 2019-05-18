#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark import SparkContext
def main(spark, train_file,val_file,test_file, train_output_file,val_output_file,test_output_file):
    df_train = spark.read.parquet(train_file)
    val = spark.read.parquet(val_file)
    test = spark.read.parquet(test_file)
    df_train.createOrReplaceTempView('train')
    val_test = val.join(test, on=['user_id'],how='full_outer')
    val_test.createOrReplaceTempView('val_test')
    train = spark.sql('''SELECT * from train WHERE user_id IN (SELECT user_id FROM val_test)''')
    train_sub = df_train.sample(False, 0.001)
    train = train.union(train_sub)
    Indexer_user = StringIndexer(inputCol="user_id", outputCol="userIndex")
    Indexer_track = StringIndexer(inputCol = "track_id", outputCol = "trackIndex")
    model_user = Indexer_user.fit(train)
    model_track = Indexer_track.fit(train).setHandleInvalid("keep")
    Indexed_train = model_user.transform(train)
    Indexed_val = model_user.transform(val)
    Indexed_test = model_user.transform(test)
    Indexed_train = model_track.transform(Indexed_train)
    Indexed_val = model_track.transform(Indexed_val)
    Indexed_test = model_track.transform(Indexed_test)
    Indexed_train.write.parquet(train_output_file)
    Indexed_val.write.parquet(val_output_file)
    Indexed_test.write.parquet(test_output_file)


    pass

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName("index").getOrCreate()

    # Get the filename from the command line
    train_file = sys.argv[1]
    val_file = sys.argv[2]
    test_file = sys.argv[3]

    # And the location to store the trained model
    train_output_file = sys.argv[4]
    val_output_file = sys.argv[5]
    test_output_file = sys.argv[6]

    # Call our main routine
    main(spark, train_file,val_file,test_file, train_output_file,val_output_file,test_output_file)

