"""
USE_SPARK = True

if USE_SPARK:
    loader = SparkDatasetLoader(
        spark_session=spark,
        path="hdfs:///data/nba.csv"
    )
else:
    loader = CsvDatasetLoader(
        path="data/nba.csv"
    )

dataset = loader.load()

for row in dataset:
    process(row)

"""