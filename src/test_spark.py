from pyspark.sql import SparkSession

def main():
    # Initialize a Spark Session
    spark = SparkSession.builder \
        .appName("SparkSetupTest") \
        .getOrCreate()

    print("✅ Spark Session created successfully!")
    print(f"   Spark Version: {spark.version}")

    # Create a sample DataFrame to test functionality
    data = [("James", "Smith", 30),
            ("Anna", "Rose", 41),
            ("Robert", "Williams", 25)]
    columns = ["firstname", "lastname", "age"]
    df = spark.createDataFrame(data, columns)

    print("✅ Sample DataFrame created and shown below:")
    df.show()

    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    main()