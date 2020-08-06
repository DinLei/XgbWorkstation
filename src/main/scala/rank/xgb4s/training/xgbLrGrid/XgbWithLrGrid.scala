package rank.xgb4s.training.xgbLrGrid

import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, VectorAssembler}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col
import rank.xgb4s.util.SparkUtil4s

object XgbWithLrGrid {
  val spark = SparkUtil4s.getSparkSession("xgb_lr")

  def geneTrainingData(hdfsPath: String): DataFrame = {
    val pathList = hdfsPath.split(",");
    assert(pathList.nonEmpty)

    var data: DataFrame = spark.read.format("libsvm").load(pathList(0))

    for (i <- 1 until pathList.length) {
      data.union(spark.read.format("libsvm").load(pathList(i)))
    }

    data
  }

  def main(args: Array[String]): Unit = {
    if (args.length != 4) {
      println("Usage: input_path,xgbModelPath,lrModelPath")
      sys.exit(1)
    }

    // 获取数据路径和存储路径
    val trainPath = args(0)
    val evalPath = args(1)
    val xgbModelPath = args(2)
    val lrModelPath = args(3)

    val lrLogPath = args(3) + ".log"
    var learningLog: String = "Training&EvalLogs:\n"

    // 获取数据：org.apache.spark.sql.DataFrame = [label: double, features: vector] ## printSchema()
    // 构造训练集和测试集
    //    val Array(train, eval) = data.randomSplit(Array(0.8, 0.2), 123)
    val train: DataFrame = geneTrainingData(trainPath)
    val eval: DataFrame = geneTrainingData(evalPath)

    // 获取参数配置文件，暂时不用
    //    val params = PropertiesUtil.getProperties("xgb_lr.properties")

    // xgboost的训练参数
    // 注意：这里先通过xgb pipeline grid寻优出最近xgb参数
    val treeNum = 100

    val xgbParam = Map(
      "eta" -> 0.1f,
      "num_round" -> treeNum,
      "max_depth" -> 6,
      "max_leaf_nodes" -> 20,
      "objective" -> "binary:logistic",
      "num_workers" -> 2,
      "missing" -> -999,
      "eval_sets" -> Map("eval" -> eval)
//      "num_class" -> 2,
    )

    // xgboost 模型
    val booster = new XGBoostClassifier(xgbParam)

    booster.setFeaturesCol("features")
    booster.setLabelCol("label")

    val xgbModel: XGBoostClassificationModel = booster.fit(train)

    // Batch prediction
    val prediction = xgbModel.transform(eval)

    // XgbModel evaluation
    val xgbEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val xgbAcc = xgbEvaluator.evaluate(prediction)

    val xgbEvaluator2 = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("prediction")
      .setMetricName("areaUnderROC")
    val xgbAuc = xgbEvaluator2.evaluate(prediction)

    learningLog += s"base_xgb model ::: accuracy is : ${xgbAcc}, auc is : ${xgbAuc}\n\n"

    //存储二进制模型文件
    //    booster.write.overwrite().save(xgbModelPath)
    xgbModel.nativeBooster.saveModel(xgbModelPath)

    // 获取预测叶子索引
    xgbModel.setLeafPredictionCol("predictLeaf")
    val xgbTrainDF = xgbModel.transform(train)
    val xgbEvalDF = xgbModel.transform(eval)

    assert(xgbTrainDF.columns.contains("predictLeaf"))
    assert(xgbEvalDF.columns.contains("predictLeaf"))

    val labelWithLeavesTrainDF = xgbTrainDF.select(
      col("label") +: col("features") +:
        (0 until treeNum).map(i => col("predictLeaf").getItem(i).as(s"leaf_$i")): _* // 变长参数
    )

    val labelWithLeavesEvalDF = xgbEvalDF.select(
      col("label") +: col("features") +:
        (0 until treeNum).map(i => col("predictLeaf").getItem(i).as(s"leaf_$i")): _* // 变长参数
    )

    // xgb模型输出 叶子节点 做onehot转换
    val encoder = new OneHotEncoderEstimator()
      .setInputCols((0 until treeNum).map(i => "leaf_" + i).toArray)
      .setOutputCols((0 until treeNum).map(i => "onehot_" + i).toArray)
      .setDropLast(false)

    val mapModel = encoder.fit(labelWithLeavesTrainDF)
    val transformedTrainDF = mapModel.transform(labelWithLeavesTrainDF)
    val transformedEvalDF = mapModel.transform(labelWithLeavesEvalDF)

    // 构造LR模型的输入样本
    val vectorAssembler = new VectorAssembler().
      setInputCols((0 until treeNum).map(i => "onehot_" + i).toArray).
      setOutputCol("lrFeatures")

    val lrTrainInput = vectorAssembler.transform(transformedTrainDF)
      .select("lrFeatures", "label")
    val lrEvalInput = vectorAssembler.transform(transformedEvalDF)
      .select("lrFeatures", "label")

    // LR 模型
    val lr: LogisticRegression = new LogisticRegression()
      .setMaxIter(1000)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setFeaturesCol("lrFeatures")
      .setLabelCol("label")


    // 逻辑回归的流水线
//    val lrPipeline = new Pipeline().setStages(Array(lr))

    val lrParamGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.2, 0.1, 0.01))
      .addGrid(lr.maxIter, Array(500, 1000, 1500))
      .addGrid(lr.elasticNetParam, Array(0.5, 0.8, 1.0))
      .build()

    // 二元分类评估

    val lrEvaluator = new BinaryClassificationEvaluator()
      .setMetricName("areaUnderROC")
      .setRawPredictionCol("prediction")
      .setLabelCol("label")

//    val cvLr = new CrossValidator()
//      .setEstimator(lrPipeline)
//      .setEvaluator(lrEvaluator)
//      .setEstimatorParamMaps(paramGrid)
//      .setNumFolds(10)  // Use 3+ in practice
//      .setParallelism(4)  // Evaluate up to 2 parameter settings in parallel
    // TV只做一次数据分割
    val tvLr = new TrainValidationSplit()
      .setEstimator(lr)
      .setEvaluator(lrEvaluator)
      .setEstimatorParamMaps(lrParamGrid)
      // 99% of the data will be used for training and the remaining 1% for validation.
      // 数据量很大不需要按7/3或8/2分
      .setTrainRatio(0.99)
      // Evaluate up to 2 parameter settings in parallel
      .setParallelism(4)

    // 训练逻辑回归模型
    val gridLrModel = tvLr.fit(lrTrainInput)

//    val bestLrModel = cvLrModel.bestModel.asInstanceOf[PipelineModel].stages(0)
//      .asInstanceOf[LogisticRegressionModel]
    val bestLrModel = gridLrModel.bestModel.asInstanceOf[LogisticRegressionModel]

    // 预测逻辑回归的值
    val predictions = bestLrModel.transform(lrEvalInput)

    // 评估模型指标之AUC
    val lrAUC = lrEvaluator.evaluate(predictions)

    // 评估模型指标之ACC
    val lrEvaluatorM = new MulticlassClassificationEvaluator().setMetricName("accuracy").setLabelCol("label")
    val lrACC = lrEvaluatorM.evaluate(predictions)

    learningLog += s"best_lr model ::: acc: ${lrACC}, auc: ${lrAUC}\n\n"
    learningLog += "The params of best lr model : " + bestLrModel.extractParamMap() + "\n"

    val lrOutcome: String = s"Coefficients: ${bestLrModel.coefficients}\nIntercept: ${bestLrModel.intercept}"
    SparkUtil4s.write2hdfs(lrModelPath, lrOutcome)

    // #.$.#.$.#.$.#.$.#.$.#.$.#.$.#.$.#.$.#.$.#.$.#.$.#.$.#.$.#.$.#.$.#.$.#.$.#.$.#.$.#.$.# //

    // 输出 LR 各个评价指标
    val trainingSummary = bestLrModel.summary

    // Obtain the objective per iteration
    val objectiveHistory = trainingSummary.objectiveHistory
    println("objectiveHistory:")
    objectiveHistory.foreach(println)

    // for multiclass, we can inspect metrics on a per-label basis
    println("False positive rate by label:")
    trainingSummary.falsePositiveRateByLabel.zipWithIndex.foreach { case (rate, label) =>
      println(s"label $label: $rate")
    }

    println("True positive rate by label:")
    trainingSummary.truePositiveRateByLabel.zipWithIndex.foreach { case (rate, label) =>
      println(s"label $label: $rate")
    }

    println("Precision by label:")
    trainingSummary.precisionByLabel.zipWithIndex.foreach { case (prec, label) =>
      println(s"label $label: $prec")
    }

    println("Recall by label:")
    trainingSummary.recallByLabel.zipWithIndex.foreach { case (rec, label) =>
      println(s"label $label: $rec")
    }

    println("F-measure by label:")
    trainingSummary.fMeasureByLabel.zipWithIndex.foreach { case (f, label) =>
      println(s"label $label: $f")
    }

    val accuracy = trainingSummary.accuracy
    val falsePositiveRate = trainingSummary.weightedFalsePositiveRate
    val truePositiveRate = trainingSummary.weightedTruePositiveRate
    val fMeasure = trainingSummary.weightedFMeasure
    val precision = trainingSummary.weightedPrecision
    val recall = trainingSummary.weightedRecall

    learningLog += "best lr model training summary: \n"
    learningLog += s"Accuracy: $accuracy, FPR: $falsePositiveRate, TPR: $truePositiveRate, " +
      s"F-measure: $fMeasure, Precision: $precision, Recall: $recall"

    SparkUtil4s.write2hdfs(lrLogPath, learningLog)
  }
}
