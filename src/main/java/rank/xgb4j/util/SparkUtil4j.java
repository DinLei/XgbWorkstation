package rank.xgb4j.util;

import org.apache.spark.sql.SparkSession;

public class SparkUtil4j {

    public static SparkSession getSparkSession(String appName) {
        return SparkSession
                .builder()
                .appName(appName)
                .config("spark.sql.warehouse.dir", "hdfs://ns1/apps/hive/warehouse/ai.db")
                .config("spark.sql.shuffle.partitions", 250)  // 3x of #cpus
                .config("spark.driver.maxResultSize", "10g")
                .enableHiveSupport()
                .getOrCreate();
    }

    public static SparkSession getSparkSession(String appName, int numPartitions) {
        return SparkSession
                .builder()
                .appName(appName)
                .config("spark.sql.warehouse.dir", "hdfs://ns1/apps/hive/warehouse/ai.db")
                .config("spark.sql.shuffle.partitions", numPartitions)  // 3x of #cpus
                .config("spark.driver.maxResultSize", "10g")
                .enableHiveSupport()
                .getOrCreate();
    }

}
