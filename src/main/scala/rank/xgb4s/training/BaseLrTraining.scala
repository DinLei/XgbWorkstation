package rank.xgb4s.training

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.sql.DataFrame
import rank.xgb4s.util.SparkUtil4s

object BaseLrTraining {
  val spark = SparkUtil4s.getSparkSession("base_lr")

  def geneTrainingData(hdfsPath: String): DataFrame = {
    val pathList = SparkUtil4s.pathRegex(hdfsPath)
    assert(pathList.nonEmpty)

    var data: DataFrame = spark.read.format("libsvm").load(pathList(0))

    for (i <- 1 until pathList.length) {
      data = data.union(spark.read.format("libsvm").load(pathList(i))).distinct()
    }

    data.distinct()
  }

  def main(args: Array[String]): Unit = {

    // 获取数据路径和存储路径
    val dataPath = args(0)
    val lrModelPath = args(1)

    val tLogPath = args(2)
    var learningLog: String = "Training&Eval-Logs:\n"

    // xgboost的训练参数
    var iterNum = 100
    var regParam = 0.3
    var elasticNetParam = 0.2

    if (args.length > 3) {
      val params: Array[String] = args(3).trim.split(",")
      if (params.length >= 1)
        iterNum = params(0).toInt
      if (params.length >= 2)
        regParam = params(1).toFloat
      if (params.length >= 3)
        elasticNetParam = params(2).toFloat
    }

    // 获取数据：org.apache.spark.sql.DataFrame = [label: double, features: vector] ## printSchema()
    // 构造训练集和测试集
    val allData: DataFrame = geneTrainingData(dataPath)
    val Array(train, eval) = allData.randomSplit(Array(0.9, 0.1), 123)

//    train.col("features").
    learningLog += s"\ntraining data schema: \n${train.schema.treeString}\n"
    val trainInfo = train.groupBy("label").count()
    val evalInfo = eval.groupBy("label").count()

    learningLog += s"train example label distribution: \n"
    trainInfo.collect().foreach(
      x => learningLog += s"label: ${x.get(0)}, count: ${x.get(1)}\n"
    )

    learningLog += s"eval example label distribution: \n"
    evalInfo.collect().foreach(
      x => learningLog += s"label: ${x.get(0)}, count: ${x.get(1)}\n"
    )

    SparkUtil4s.write2hdfs(tLogPath, learningLog)

    // LR 模型
    val lr: LogisticRegression = new LogisticRegression()
      .setMaxIter(iterNum)
      .setRegParam(regParam)
      .setElasticNetParam(elasticNetParam)
      .setFeaturesCol("features")
      .setLabelCol("label")

    // 训练逻辑回归模型
    val lrModel = lr.fit(train)

    // 预测逻辑回归的值
    val lrPred = lrModel.transform(eval)

    // 二元分类评估
    val lrEvaluator = new BinaryClassificationEvaluator()
      .setMetricName("areaUnderROC")
      .setRawPredictionCol("probability")
      .setLabelCol("label")

    learningLog += s"\nlrPred schema: \n${lrPred.schema.treeString}\n"
    lrPred.filter("label == 0").head(5).foreach(
      row => {
        learningLog += s"${row.toString()}\n"
      }
    )
    lrPred.filter("label == 1").head(5).foreach(
      row => {
        learningLog += s"${row.toString()}\n"
      }
    )

    // 评估模型指标之AUC
    val lrAUC = lrEvaluator.evaluate(lrPred)

    // 评估模型指标之ACC
    val lrEvaluator2 = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
      .setLabelCol("label")
    val lrACC = lrEvaluator2.evaluate(lrPred)

    learningLog += s"\nbase_lr model ::: acc: ${lrACC}, auc: ${lrAUC}\n"

    SparkUtil4s.saveLrModel(lrModel, lrModelPath)

    // #.$.#.$.#.$.#.$.#.$.#.$.#.$.#.$.#.$.#.$.#.$.#.$.#.$.#.$.#.$.#.$.#.$.#.$.#.$.#.$.#.$.# //
    SparkUtil4s.write2hdfs(tLogPath, learningLog)
  }
}
