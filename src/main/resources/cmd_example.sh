#./bin/spark-submit \
#  --class <main-class> \        # 应用程序主入口类
#  --master <master-url> \       # 集群的 Master Url
#  --deploy-mode <deploy-mode> \ # 部署模式
#  --conf <key>=<value> \        # 可选配置
#  ... # other options
#  <application-jar> \           # Jar 包路径
#  [application-arguments]       #传递给主入口类的参数

# 在集群环境下，application-jar 必须能被集群中所有节点都能访问，
# 可以是 HDFS 上的路径；也可以是本地文件系统路径，
# 如果是本地文件系统路径，则要求集群中每一个机器节点上的相同路径都存在该 Jar 包。

# 本地模式提交应用
spark-submit \
--class org.apache.spark.examples.SparkPi \
--master local[2] \
/usr/app/spark-2.4.0-bin-hadoop2.6/examples/jars/spark-examples_2.11-2.4.0.jar \
100   # 传给 SparkPi 的参数


# 以client模式提交到standalone集群
spark-submit \
--class org.apache.spark.examples.SparkPi \
--master spark://hadoop001:7077 \
--executor-memory 2G \
--total-executor-cores 10 \
/usr/app/spark-2.4.0-bin-hadoop2.6/examples/jars/spark-examples_2.11-2.4.0.jar \
100

# 以cluster模式提交到standalone集群
spark-submit \
--class org.apache.spark.examples.SparkPi \
--master spark://207.184.161.138:7077 \
--deploy-mode cluster \
--supervise \  # 配置此参数代表开启监督，如果主应用程序异常退出，则自动重启 Driver
--executor-memory 2G \
--total-executor-cores 10 \
/usr/app/spark-2.4.0-bin-hadoop2.6/examples/jars/spark-examples_2.11-2.4.0.jar \
100
