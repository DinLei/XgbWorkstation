package rank.xgb4s.training

import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{Column, DataFrame, Row}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.StringType
import rank.xgb4s.util.{SparkUtil4s, XgbUtils}
import rank.xgb4s.util.metrics.{MetricsFun4spark, NormalizedEntropy}

import scala.collection.mutable

object XgbLrTraining {
  val spark = SparkUtil4s.getSparkSession("xgb_lr")

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
    if (args.length < 5) {
      sys.exit(1)
    }

    // 获取数据路径和存储路径
    val trainPath = args(0)
    val testPath = args(1)
    val xgbModelPath = args(2)
    val lrModelPath = args(3)

    val tLogPath = args(4)
    var learningLog: String = "Training&Eval-Logs:\n"

    // xgboost的参数
    val treeNum = 100
    var xgbParam = Map(
      "objective" -> "binary:logistic",
      "num_workers" -> 100,
      "num_round" -> treeNum,
      "max_depth" -> 8,
      "subsample" -> 0.8,
      "colsample_bytree" -> 0.8,
      "min_child_weight" -> 2,
      "scale_pos_weight" -> 1.0,
      "lambda" -> 1,
      "alpha" -> 0,
      "gamma" -> 0.2,
      "eta" -> 0.1
    )
    var iterNum = 100
    var regParam = 0.3
    var elasticNetParam = 0.2

    if(args.length > 5) {
      xgbParam = xgbParam ++ XgbUtils.paramsParse(args(5))
    }
    if(args.length > 6) {
      val params: Array[String] = args(3).trim.split(",")
      iterNum = params(0).toInt
      regParam = params(1).toFloat
      elasticNetParam = params(2).toFloat
    }
    // 获取数据：org.apache.spark.sql.DataFrame = [label: double, features: vector] ## printSchema()
    // 构造训练集和测试集

    val allData: DataFrame = geneTrainingData(trainPath)
    val Array(train, eval) = allData.randomSplit(Array(0.99, 0.01), 123)
    val test: DataFrame = geneTrainingData(testPath)

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
    watches += "eval" -> eval

    // xgboost 模型
    val booster = new XGBoostClassifier(xgbParam)
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setEvalSets(watches.toMap)
      .setCustomEval(new NormalizedEntropy)
    //      .setMaximizeEvaluationMetrics(false)

    val xgbModel: XGBoostClassificationModel = booster.fit(train)

    // Batch prediction
    xgbModel.setLeafPredictionCol("predictLeaf")
    val xgbPredTr = xgbModel.transform(train)
    val xgbPredE = xgbModel.transform(eval)
    val xgbPredTe = xgbModel.transform(test)

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
    learningLog += s"\nxgboost in eval:  auc = ${xgbAucE} ,  ne = ${xgbNeE} ,  calibration = ${xgbCaE} ,  logloss = ${xgbLlsE}\n"
    learningLog += s"\nxgboost in test:  auc = ${xgbAucTe} , ne = ${xgbNeTe} , calibration = ${xgbCaTe} , logloss = ${xgbLlsTe}\n"

    learningLog += s"\nXGBoostClassificationModel summary: ${xgbModel.summary}\n\n"

    //存储二进制模型文件
    //    booster.write.overwrite().save(xgbModelPath)
    xgbModel.nativeBooster.saveModel(xgbModelPath)
    SparkUtil4s.write2hdfs(tLogPath, learningLog)

    // 获取预测叶子索引
    val labelWithLeavesTrainDF = xgbPredTr.select(
      col("label") +: col("features") +:
        (0 until treeNum).map(i => col("predictLeaf").getItem(i).as(s"leaf_$i").cast(StringType)): _* // 变长参数
    )

    val labelWithLeavesEvalDF = xgbPredE.select(
      col("label") +: col("features") +:
        (0 until treeNum).map(i => col("predictLeaf").getItem(i).as(s"leaf_$i").cast(StringType)): _* // 变长参数
    )

    val labelWithLeavesTestDF = xgbPredTe.select(
      col("label") +: col("features") +:
        (0 until treeNum).map(i => col("predictLeaf").getItem(i).as(s"leaf_$i").cast(StringType)): _* // 变长参数
    )

    val leafIndexers = (0 until treeNum).map(i =>s"leaf_$i").map(
      col => {
        new StringIndexer().setInputCol(col).setOutputCol(col + "_idx")
      }
    ).toArray

    val leafIndexersPipe = new Pipeline().setStages(leafIndexers).fit(labelWithLeavesTrainDF)

    val labelWithLeavesTrainDF1 = leafIndexersPipe.transform(labelWithLeavesTrainDF)
    val labelWithLeavesEvalDF1 = leafIndexersPipe.transform(labelWithLeavesEvalDF)
    val labelWithLeavesTestDF1 = leafIndexersPipe.transform(labelWithLeavesTestDF)

    learningLog += s"\nlabelWithLeavesTrainDF schema: \n${labelWithLeavesTrainDF1.schema.treeString}\n"
    learningLog += s"\nlabelWithLeavesTrainDF: \n${labelWithLeavesTrainDF1.head().toString()}\n"
//    learningLog += s"\nlabelWithLeavesEvalDF schema: \n${labelWithLeavesEvalDF1.schema.treeString}\n"
//    learningLog += s"\nlabelWithLeavesEvalDF: \n${labelWithLeavesEvalDF1.head().toString()}\n"

    // xgb模型输出 叶子节点 做onehot转换
    val encoder = new OneHotEncoderEstimator()
      .setInputCols((0 until treeNum).map(i => s"leaf_${i}_idx").toArray)
      .setOutputCols((0 until treeNum).map(i => s"onehot_$i").toArray)
      .setDropLast(false)

    val mapModel = encoder.fit(labelWithLeavesTrainDF1)
    val transformedTrainDF = mapModel.transform(labelWithLeavesTrainDF1)
    val transformedEvalDF = mapModel.transform(labelWithLeavesEvalDF1)
    val transformedTestDF = mapModel.transform(labelWithLeavesTestDF1)

    learningLog += s"\ntransformedTrainDF schema: \n${transformedTrainDF.schema.treeString}\n"
    learningLog += s"\ntransformedTrainDF: \n${transformedTrainDF.head().toString()}\n"
//    learningLog += s"\ntransformedEvalDF schema: \n${transformedEvalDF.schema.treeString}\n"
//    learningLog += s"\ntransformedEvalDF: \n${transformedEvalDF.head().toString()}\n"

    // 构造LR模型的输入样本
    val vectorAssembler = new VectorAssembler().
      setInputCols((0 until treeNum).map(i => "onehot_" + i).toArray).
      setOutputCol("lrFeatures")

    val lrTrainInput = vectorAssembler.transform(transformedTrainDF)
      .select("lrFeatures", "label")
    val lrEvalInput = vectorAssembler.transform(transformedEvalDF)
      .select("lrFeatures", "label")
    val lrTestInput = vectorAssembler.transform(transformedTestDF)
      .select("lrFeatures", "label")

    learningLog += s"\nlrTrainInput schema: \n${lrTrainInput.schema.treeString}\n"
    learningLog += s"\nlrTrainInput: \n${lrTrainInput.head().toString()}\n"
//    learningLog += s"\nlrEvalInput schema: \n${lrEvalInput.schema.treeString}\n"
//    learningLog += s"\nlrEvalInput: \n${lrEvalInput.head().toString()}\n"

    // LR 模型
    val lr: LogisticRegression = new LogisticRegression()
      .setMaxIter(iterNum)
      .setRegParam(regParam)
      .setElasticNetParam(elasticNetParam)
      .setFeaturesCol("lrFeatures")
      .setLabelCol("label")

    // 训练逻辑回归模型
    val lrModel = lr.fit(lrTrainInput)

    // 二元分类评估

    val lrEvaluator = new BinaryClassificationEvaluator()
      .setMetricName("areaUnderROC")
      .setRawPredictionCol("probability")
      .setLabelCol("label")

    // 预测逻辑回归的值
    val lrPredTr = lrModel.transform(lrTrainInput)
    val lrPredE = lrModel.transform(lrEvalInput)
    val lrPredTe = lrModel.transform(lrTestInput)

    // 评估模型指标之AUC
    val lrAUCTr = lrEvaluator.evaluate(lrPredTr)
    val lrAUCE = lrEvaluator.evaluate(lrPredE)
    val lrAUCTe = lrEvaluator.evaluate(lrPredTe)

    val lrNeTr = MetricsFun4spark.normalizedEntropy(lrPredTr, "label", "probability")
    val lrNeE = MetricsFun4spark.normalizedEntropy(lrPredE, "label", "probability")
    val lrNeTe = MetricsFun4spark.normalizedEntropy(lrPredTe, "label", "probability")

    val lrCaTr = MetricsFun4spark.calibration(lrPredTr, "label", "probability")
    val lrCaE = MetricsFun4spark.calibration(lrPredE, "label", "probability")
    val lrCaTe = MetricsFun4spark.calibration(lrPredTe, "label", "probability")

    val lrLlsTr = MetricsFun4spark.logLoss(lrPredTr, "label", "probability")
    val lrLlsE = MetricsFun4spark.logLoss(lrPredE, "label", "probability")
    val lrLlsTe = MetricsFun4spark.logLoss(lrPredTe, "label", "probability")

    learningLog += s"\nlr in train: auc = ${lrAUCTr} , ne = ${lrNeTr} , calibration = ${lrCaTr} , logloss = ${lrLlsTr}\n"
    learningLog += s"\nlr in eval:  auc = ${lrAUCE} ,  ne = ${lrNeE} ,  calibration = ${lrCaE} ,  logloss = ${lrLlsE}\n"
    learningLog += s"\nlr in test:  auc = ${lrAUCTe} , ne = ${lrNeTe} , calibration = ${lrCaTe} , logloss = ${lrLlsTe}\n"

    SparkUtil4s.saveLrModel(lrModel, lrModelPath)
    SparkUtil4s.write2hdfs(tLogPath, learningLog)
  }
}
