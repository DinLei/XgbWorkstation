package rank.xgb4s.training.xgbLrGrid

import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.DataFrame
import rank.xgb4s.util.SparkUtil4s

object XgbGrid {
  val spark = SparkUtil4s.getSparkSession("xgb_lr")

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
    if (args.length != 3) {
      println("Usage: input_path,xgbModelPath,lrModelPath")
      sys.exit(1)
    }

    // 获取数据路径和存储路径
    val trainPath = args(0)
    val testPath = args(1)
    val nativeModelPath = args(2)
    val logPath = args(2) + ".log"

    var learningLog: String = "Training&EvalLogs:\n"

    // 获取数据：org.apache.spark.sql.DataFrame = [label: double, features: vector] ## printSchema()
    // 构造训练集和测试集
    val data: DataFrame = geneTrainingData(trainPath)
    val Array(train, eval) = data.randomSplit(Array(0.99, 0.01), 123)

    val test: DataFrame = geneTrainingData(testPath)

    // 获取参数配置文件，暂时不用
    //    val params = PropertiesUtil.getProperties("xgb_lr.properties")

    // xgboost的训练参数

    val xgbParam = Map(
      "eta" -> 0.1f,
      "num_round" -> 50,
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

    val evaluator2 = new BinaryClassificationEvaluator()
        .setLabelCol("label")
        .setRawPredictionCol("probability")
        .setMetricName("areaUnderROC")

    // Tune model using cross validation
    // 注意：这里整体寻优计算量太大，分步寻优
    val xgbParamGrid = new ParamGridBuilder()
      .addGrid(booster.maxDepth, 6 until 10)
      .addGrid(booster.numRound, Array(50, 100, 150))
      .addGrid(booster.eta, 0.05 until (0.21, 0.05))
      .build()

//    val cv = new CrossValidator()
//      .setEstimator(pipeline)
//      .setEvaluator(evaluator)
//      .setEstimatorParamMaps(paramGrid)
//      .setNumFolds(10)

    val tv = new TrainValidationSplit()
      .setEstimator(booster)
      .setEvaluator(evaluator2)
      .setEstimatorParamMaps(xgbParamGrid)
      // 80% of the data will be used for training and the remaining 20% for validation.
      .setTrainRatio(0.99)
      // Evaluate up to 2 parameter settings in parallel
      .setParallelism(4)

    val gridModel = tv.fit(train)

    val bestModel = gridModel.bestModel.asInstanceOf[XGBoostClassificationModel]

    learningLog += "The params of best XGBoostClassification model : " + bestModel.extractParamMap() + "\n"
    learningLog += "The training summary of best XGBoostClassificationModel : " +  bestModel.summary + "\n"

    // Batch prediction
    val xgbPred = bestModel.transform(test)
    val xgbEvaluator2 = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("probability")
      .setMetricName("areaUnderROC")
    val xgbAuc = xgbEvaluator2.evaluate(xgbPred)

    learningLog += s"base_xgb model in test auc is : ${xgbAuc}\n\n"

    // Export the XGBoostClassificationModel as local XGBoost model,
    // then you can load it back in local Python environment.
    bestModel.nativeBooster.saveModel(nativeModelPath)
    SparkUtil4s.write2hdfs(logPath, learningLog)
  }
}
