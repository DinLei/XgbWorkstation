package rank.xgb4s.training

import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{Column, DataFrame, Row}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.StringType
import rank.xgb4s.util.SparkUtil4s

object XgbLrTraining {
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

    // xgboost的训练参数
    var treeNum = 100
    var treeMaxDepth = 3
    var treeMaxLeaves = 10
    var nWorks = 10
    if( args.length > 5) {
      val params: Array[String] = args(5).trim.split(",")
      if(params.length >= 1)
        treeNum = params(0).toInt
      if(params.length >= 2)
        treeMaxDepth = params(1).toInt
      if(params.length >= 3)
        treeMaxLeaves = params(2).toInt
      if(params.length >= 4)
        nWorks = params(3).toInt
    }

    // 获取数据：org.apache.spark.sql.DataFrame = [label: double, features: vector] ## printSchema()
    // 构造训练集和测试集

    val allData: DataFrame = geneTrainingData(trainPath)
    val Array(train, eval) = allData.randomSplit(Array(0.99, 0.01), 123)

    var testData: DataFrame = null
    if(testPath.length > 1) {
      testData = geneTrainingData(testPath)
    }

    val trainInfo = train.groupBy("label").count()
    val evalInfo = eval.groupBy("label").count()

    learningLog += s"train example label distribution: \n"
    trainInfo.collect().foreach(
      x=> learningLog += s"label: ${x.get(0)}, count: ${x.get(1)}\n"
    )

    learningLog += s"eval example label distribution: \n"
    evalInfo.collect().foreach(
      x=> learningLog += s"label: ${x.get(0)}, count: ${x.get(1)}\n"
    )

    if(testData != null) {
      val testInfo = testData.groupBy("label").count()
      learningLog += s"test example label distribution: \n"
      testInfo.collect().foreach(
        x=> learningLog += s"label: ${x.get(0)}, count: ${x.get(1)}\n"
      )
    }

    // 获取参数配置文件，暂时不用
    //    val params = PropertiesUtil.getProperties("xgb_lr.properties")

    val xgbParam = Map(
      "eta" -> 0.1f,
      "num_round" -> treeNum,
      "max_depth" -> treeMaxDepth,
      "max_leaf_nodes" -> treeMaxLeaves,
      "objective" -> "binary:logistic",
      "num_workers" -> nWorks,
//      "missing" -> -999,
      "eval_sets" -> Map("eval" -> eval),
      "eval_metric" -> "auc"
//      "num_class" -> 2,
    )

    // xgboost 模型
    val booster = new XGBoostClassifier(xgbParam)
      .setFeaturesCol("features")
      .setLabelCol("label")

    val xgbModel: XGBoostClassificationModel = booster.fit(train)

    // Batch prediction
    val xgbPred = xgbModel.transform(eval)

    learningLog += s"\nxgbPred schema: \n${xgbPred.schema.treeString}\n"
    xgbPred.head(5).foreach(
      row => {
        learningLog += s"${row.toString()}\n"
      }
    )

    // XgbModel evaluation
    val xgbEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val xgbAcc = xgbEvaluator.evaluate(xgbPred)

    val xgbEvaluator2 = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("probability")
      .setMetricName("areaUnderROC")
    val xgbAuc = xgbEvaluator2.evaluate(xgbPred)

    learningLog += s"base_xgb model in eval ::: accuracy is : ${xgbAcc}, auc is : ${xgbAuc}\n\n"

    if(testData != null) {
      val xgbPredT = xgbModel.transform(testData)
      val xgbEvaluator2T = new BinaryClassificationEvaluator()
        .setLabelCol("label")
        .setRawPredictionCol("probability")
        .setMetricName("areaUnderROC")
      val xgbAucT = xgbEvaluator2T.evaluate(xgbPredT)

      learningLog += s"base_xgb model in test ::: auc is : ${xgbAucT}\n\n"
    }

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
        (0 until treeNum).map(i => col("predictLeaf").getItem(i).as(s"leaf_$i").cast(StringType)): _* // 变长参数
    )

    val labelWithLeavesEvalDF = xgbEvalDF.select(
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

    learningLog += s"\nlrTrainInput schema: \n${lrTrainInput.schema.treeString}\n"
    learningLog += s"\nlrTrainInput: \n${lrTrainInput.head().toString()}\n"
//    learningLog += s"\nlrEvalInput schema: \n${lrEvalInput.schema.treeString}\n"
//    learningLog += s"\nlrEvalInput: \n${lrEvalInput.head().toString()}\n"

    // LR 模型
    val lr: LogisticRegression = new LogisticRegression()
      .setMaxIter(500)
      .setRegParam(0.3)
      .setElasticNetParam(0.2)
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
    val lrPred = lrModel.transform(lrEvalInput)

    // 评估模型指标之AUC
    val lrAUC = lrEvaluator.evaluate(lrPred)

    // 评估模型指标之ACC
    val lrEvaluator2 = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
      .setLabelCol("label")
    val lrACC = lrEvaluator2.evaluate(lrPred)

    learningLog += s"\nbase_lr model ::: acc: ${lrACC}, auc: ${lrAUC}\n\n"

    SparkUtil4s.saveLrModel(lrModel, lrModelPath)
    SparkUtil4s.write2hdfs(tLogPath, learningLog)
  }
}
