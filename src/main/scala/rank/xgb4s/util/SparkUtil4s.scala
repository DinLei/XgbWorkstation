package rank.xgb4s.util


import java.io.{BufferedWriter, FileOutputStream, ObjectOutputStream, OutputStreamWriter}
import scala.math._
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FSDataOutputStream, FileSystem, Path}
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ListBuffer

object SparkUtil4s {

  /*
  val appName: String = "nc-universal-rec"
  val configEnv: Config = ConfigFactory.load("events.conf")

  val tableMaster: Option[String] = Some(configEnv.getString(s"${appName}.table-master"))
  val colFamily: Option[String] = Some(configEnv.getString(s"${appName}.column-family"))
  val hBaseHost: Option[String] = Some(configEnv.getString(s"${appName}.hbase-db-host"))
  val hBasTable: Option[String] = Some(configEnv.getString(s"${appName}.hbase-table"))
  val hiveDataWarehouse: Option[String] = Some(configEnv.getString(s"${appName}.hive-data-warehouse"))

  val defaultPartitions: Option[Int] = Some(configEnv.getInt(s"${appName}.defaultPartitions"))
  val ccoModelSavePath: Option[String] = Some(configEnv.getString(s"${appName}.ccoModelSavePath"))
  */

  val defaultPartitions: Option[Int] = Some(250)


  def getSparkSession(appName: String): SparkSession = {
    SparkSession
      .builder()
      .appName(appName)
      .config("hive.exec.dynamic.partition", "true")
      .config("hive.exec.dynamic.partition.mode", "nonstrict")
//      .config("spark.sql.warehouse.dir", hiveDataWarehouse.get)
//      .config("spark.sql.shuffle.partitions", defaultPartitions.get)
      .config("spark.driver.maxResultSize", "10g")
      .enableHiveSupport()
      .getOrCreate()
  }

  def getSparkContext(appName: String): SparkContext = {
    val conf = new SparkConf().setAppName(appName)
//    conf.registerKryoClasses(
//      Array(
//        classOf[RandomAccessSparseVector]
//      )
//    )
    new SparkContext(conf)
  }

  def readDataFromHive(hql:String): DataFrame = {
    val spark: SparkSession = SparkUtil4s.getSparkSession("sparkOnHive")
    spark.sql(hql)
  }


  def write2hdfs(hdfsPath: String, content: String): Unit = {
    val path: Path = new Path(hdfsPath)
    val fs: FileSystem = FileSystem.get( new Configuration())
    if(fs.exists(path))
      fs.delete(path, true)
    val dataOutputStream: FSDataOutputStream = fs.create(path)
    val bw: BufferedWriter = new BufferedWriter(new OutputStreamWriter(dataOutputStream, "UTF-8"))
    bw.write(content)
    bw.close()
  }

  def saveLrModel(lrModel: LogisticRegressionModel, model_path: String): Unit = {
    // 保存模型文件 obj
    val lrOutcome: String = s"Coefficients: ${lrModel.coefficients}\nIntercept: ${lrModel.intercept}"
    write2hdfs(model_path, lrOutcome)
  }

  // 临时的简单解析路径函数
  def pathRegex(path: String): Array[String] = {
    var left = path.indexOf('[', 0)
    var right = -1
    if(left >= 0) {
      right = path.indexOf(']', left)
      if(right > left) {
        val ans:ListBuffer[String] = ListBuffer()
        val f_val = path.substring(left+1, right)
        val Array(start, end) = f_val.split("-")
        for(i <- start.toInt until end.toInt+1) {
          ans.+=(path.substring(0, left)+i+path.substring(right+1))
        }
        return ans.toArray
      }
    }
    Array(path)
  }

}