package rank.xgb4s.training

import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.sql.DataFrame
import rank.xgb4s.util.{SparkUtil4s, XgbUtils}
import rank.xgb4s.util.metrics.{Calibration, NormalizedEntropy}
import rank.xgb4s.util.metrics.MetricsFun4spark

import scala.collection.mutable

object BaseXgbTraining {
  val spark = SparkUtil4s.getSparkSession("xgboost")

  def geneTrainingData(hdfsPath: String): DataFrame = {
    val pathList = SparkUtil4s.pathRegex(hdfsPath)
    assert(pathList.nonEmpty)

    var data: DataFrame = spark.read.format("libsvm").load(pathList(0))

    for (i <- 1 until pathList.length) {
      data = data.union(spark.read.format("libsvm").load(pathList(i)))
    }

    data
  }

  def main(args: Array[String]): Unit = {
    if (args.length < 4) {
      sys.exit(1)
    }

    // 获取数据路径和存储路径
    val trainPath = args(0)
    val testPath = args(1)
    val xgbModelPath = args(2)

    val tLogPath = args(3)
    var learningLog: String = "Training&Eval-Logs:\n"

    // xgboost的训练参数
    var xgbParam = Map(
      "objective" -> "binary:logistic",
      "num_workers" -> 80,
      "num_round" -> 100,
      "max_depth" -> 8,
      "subsample" -> 0.8,
      "colsample_bytree" -> 0.8,
      "min_child_weight" -> 2,
      "scale_pos_weight" -> 1,
      "lambda" -> 1,
      "alpha" -> 0,
      "gamma" -> 0.2,
      "eta" -> 0.1
      //      "eval_metric" -> "logloss"
      //      "num_class" -> 2,
      //      "missing" -> -999,
    )

    if( args.length > 4) {
      xgbParam = xgbParam ++ XgbUtils.paramsParse(args(4))
    }

    // 获取数据：org.apache.spark.sql.DataFrame = [label: double, features: vector] ## printSchema()
    // 构造训练集和测试集

    val allData: DataFrame = geneTrainingData(trainPath)
    val Array(train, eval) = allData.randomSplit(Array(0.99, 0.01), 123)
    val test = geneTrainingData(testPath)

    val trainInfo = train.groupBy("label").count()
    val evalInfo = eval.groupBy("label").count()
    val testInfo = test.groupBy("label").count()

    learningLog += s"train example label distribution: \n"
    trainInfo.collect().foreach(
      x=> learningLog += s"label: ${x.get(0)}, count: ${x.get(1)}\n"
    )

    learningLog += s"eval example label distribution: \n"
    evalInfo.collect().foreach(
      x=> learningLog += s"label: ${x.get(0)}, count: ${x.get(1)}\n"
    )

    learningLog += s"test example label distribution: \n"
    testInfo.collect().foreach(
      x=> learningLog += s"label: ${x.get(0)}, count: ${x.get(1)}\n"
    )

    // 获取参数配置文件，暂时不用
    //    val params = PropertiesUtil.getProperties("xgb_lr.properties")

    val watches = new mutable.HashMap[String, DataFrame]
//    watches += "train" -> train
    watches += "eval" -> eval
//    watches += "test" -> test

    // xgboost 模型
    val booster = new XGBoostClassifier(xgbParam)
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setEvalSets(watches.toMap)
//      .setEvalMetric("auc")
//      .setCustomEval(new Calibration)
      .setCustomEval(new NormalizedEntropy)
//      .setMaximizeEvaluationMetrics(false)

    val xgbModel: XGBoostClassificationModel = booster.fit(train)

    val featureScoreMap = xgbModel.nativeBooster.getFeatureScore()
//    xgbModel.nativeBooster.getScore("", "gain")
    val sortedScoreMap = featureScoreMap.toSeq.sortBy(-_._2) // descending order
    learningLog += s"\nxgboost model features importance: \n${sortedScoreMap}\n"

    // Batch prediction
    val xgbPredTr = xgbModel.transform(train)
    val xgbPredE = xgbModel.transform(eval)
    val xgbPredTe = xgbModel.transform(test)

    learningLog += s"\nxgbPred schema: \n${xgbPredTe.schema.treeString}\n"
    xgbPredTe.head(2).foreach(
      row => {
        learningLog += s"${row.toString()}\n"
      }
    )

    // XgbModel evaluation
    val xgbEvaluator2 = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("probability")
      .setMetricName("areaUnderROC")

    val xgbAucTr = xgbEvaluator2.evaluate(xgbPredTr)
    val xgbAucE = xgbEvaluator2.evaluate(xgbPredE)
    val xgbAucTe = xgbEvaluator2.evaluate(xgbPredTe)

    val xgbNeTr = MetricsFun4spark.normalizedEntropy(xgbPredTr, "label", "probability")
    val xgbNeE = MetricsFun4spark.normalizedEntropy(xgbPredE, "label", "probability")
    val xgbNeTe = MetricsFun4spark.normalizedEntropy(xgbPredTe, "label", "probability")

    val xgbCaTr = MetricsFun4spark.calibration(xgbPredTr, "label", "probability")
    val xgbCaE = MetricsFun4spark.calibration(xgbPredE, "label", "probability")
    val xgbCaTe = MetricsFun4spark.calibration(xgbPredTe, "label", "probability")

    val xgbLlsTr = MetricsFun4spark.logLoss(xgbPredTr, "label", "probability")
    val xgbLlsE = MetricsFun4spark.logLoss(xgbPredE, "label", "probability")
    val xgbLlsTe = MetricsFun4spark.logLoss(xgbPredTe, "label", "probability")

    learningLog += s"\nxgboost in train: auc = ${xgbAucTr} , ne = ${xgbNeTr} , calibration = ${xgbCaTr} , logloss = ${xgbLlsTr}\n"
    learningLog += s"\nxgboost in eval:  auc = ${xgbAucE} , ne = ${xgbNeE} , calibration = ${xgbCaE} , logloss = ${xgbLlsE}\n"
    learningLog += s"\nxgboost in test:  auc = ${xgbAucTe} , ne = ${xgbNeTe} , calibration = ${xgbCaTe} , logloss = ${xgbLlsTe}\n"

    learningLog += s"\nXGBoostClassificationModel summary: ${xgbModel.summary}\n\n"

    //存储二进制模型文件
    //    booster.write.overwrite().save(xgbModelPath)
    xgbModel.nativeBooster.saveModel(xgbModelPath)
    SparkUtil4s.write2hdfs(tLogPath, learningLog)
  }
}
