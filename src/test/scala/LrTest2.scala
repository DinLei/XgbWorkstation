import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import rank.xgb4s.util.SparkUtil4s

object LrTest2 {

  val spark = SparkUtil4s.getSparkSession("lr_test")

  def main(args: Array[String]): Unit = {
    // Prepare training and test data.
    val trainPath = args(0)
    val data = spark.read.format("libsvm").load(trainPath)
    val Array(training, test) = data.randomSplit(Array(0.9, 0.1), seed = 12345)

//    val lr = new LinearRegression()
    val lr = new LogisticRegression()
      .setMaxIter(10)

    // We use a ParamGridBuilder to construct a grid of parameters to search over.
    // TrainValidationSplit will try all combinations of values and determine best model using
    // the evaluator.
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .addGrid(lr.fitIntercept)
      .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
      .build()

    // In this case the estimator is simply the linear regression.
    // A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(lr)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      // 80% of the data will be used for training and the remaining 20% for validation.
      .setTrainRatio(0.8)
      // Evaluate up to 2 parameter settings in parallel
      .setParallelism(2)

    // Run train validation split, and choose the best set of parameters.
    val model = trainValidationSplit.fit(training)

    // Make predictions on test data. model is the model with combination of parameters
    // that performed best.
    model.transform(test)
      .select("features", "label", "prediction")
      .show()

    val bestLrModel = model.bestModel.asInstanceOf[LogisticRegressionModel]

    // 输出 LR 各个评价指标
    val trainingSummary = bestLrModel.summary
    var learningLog = ""

    // Obtain the objective per iteration
    val objectiveHistory = trainingSummary.objectiveHistory
    println("objectiveHistory:")
    objectiveHistory.foreach(println)

    // for multiclass, we can inspect metrics on a per-label basis
    learningLog += "False positive rate by label:\n"
    trainingSummary.falsePositiveRateByLabel.zipWithIndex.foreach { case (rate, label) =>
      learningLog += s"label $label: $rate\n"
    }

    learningLog += "True positive rate by label:\n"
    trainingSummary.truePositiveRateByLabel.zipWithIndex.foreach { case (rate, label) =>
      learningLog += s"label $label: $rate\n"
    }

    learningLog += "Precision by label:\n"
    trainingSummary.precisionByLabel.zipWithIndex.foreach { case (prec, label) =>
      learningLog += s"label $label: $prec\n"
    }

    learningLog += "Recall by label:\n"
    trainingSummary.recallByLabel.zipWithIndex.foreach { case (rec, label) =>
      learningLog += s"label $label: $rec\n"
    }


    learningLog += "F-measure by label:\n"
    trainingSummary.fMeasureByLabel.zipWithIndex.foreach { case (f, label) =>
      learningLog += s"label $label: $f\n"
    }

    val auc = trainingSummary.asBinary.areaUnderROC
    val accuracy = trainingSummary.accuracy
    val falsePositiveRate = trainingSummary.weightedFalsePositiveRate
    val truePositiveRate = trainingSummary.weightedTruePositiveRate
    val fMeasure = trainingSummary.weightedFMeasure
    val precision = trainingSummary.weightedPrecision
    val recall = trainingSummary.weightedRecall

    learningLog += "lr model training summary: \n"
    learningLog += s"Accuracy: $accuracy, FPR: $falsePositiveRate, TPR: $truePositiveRate,\n" +
      s"F-measure: $fMeasure, Precision: $precision, Recall: $recall, AUC: ${auc}"

    bestLrModel.numFeatures
    printf(s"bestLrModel.numClasses: ${bestLrModel.numClasses}, ${bestLrModel.numFeatures}")
    printf(s"bestLrModel.coefficientMatrix: ${bestLrModel.coefficientMatrix}\n")
    printf(s"bestLrModel.coefficients: ${bestLrModel.coefficients}]\n")
    printf(s"bestLrModel.interceptVector: ${bestLrModel.interceptVector}\n")
    printf(s"bestLrModel.params: ${bestLrModel.params}\n")
    printf(s"bestLrModel.extractParamMap: ${bestLrModel.extractParamMap()}\n")

  }
}
