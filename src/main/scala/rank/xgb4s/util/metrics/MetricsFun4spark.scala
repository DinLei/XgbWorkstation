package rank.xgb4s.util.metrics

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType

import scala.math.log

object MetricsFun4spark {

  def normalizedEntropy(dataset: DataFrame, labelCol: String, predLabel: String): Double = {
    val scoreAndLabels =
      dataset.select(col(predLabel), col(labelCol).cast(DoubleType)).rdd.map {
        case Row(rawPrediction: Vector, label: Double) => Row(rawPrediction(1), label)
        case Row(rawPrediction: Double, label: Double) => Row(rawPrediction, label)
      }

    val nRow = dataset.count().toDouble
    val pLogLoss = -1.0d / nRow * scoreAndLabels.map {
      case Row(score: Double, label: Double) =>
        label * log(score) + (1 - label) * log(1 - score)
    }.reduce(_ + _)

    val posNum = scoreAndLabels.map {
      case Row(score: Double, label: Double) => label
    }.reduce(_ + _)

    val avgProb = posNum / nRow
    val trueLoss = -1.0d * (avgProb * log(avgProb) + (1 - avgProb) * log(1 - avgProb))

    pLogLoss / trueLoss
  }

  def calibration(dataset: DataFrame, labelCol: String, predLabel: String): Double = {
    val scoreAndLabels =
      dataset.select(col(predLabel), col(labelCol).cast(DoubleType)).rdd.map {
        case Row(rawPrediction: Vector, label: Double) => (rawPrediction(1), label)
        case Row(rawPrediction: Double, label: Double) => (rawPrediction, label)
      }

    val predSum = scoreAndLabels.map {
      sl => sl._1
    }.reduce(_ + _)

    val posNum = scoreAndLabels.map {
      sl => sl._2
    }.reduce(_ + _)

    predSum / posNum
  }

  def logLoss(dataset: DataFrame, labelCol: String, predLabel: String): Double = {
    val scoreAndLabels =
      dataset.select(col(predLabel), col(labelCol).cast(DoubleType)).rdd.map {
        case Row(rawPrediction: Vector, label: Double) => (rawPrediction(1), label)
        case Row(rawPrediction: Double, label: Double) => (rawPrediction, label)
      }

    val nRow = dataset.count().toDouble
    val pLogLoss = -1.0d / nRow * scoreAndLabels.map {
      //      case Row(score: Double, label: Double) => label * log(score) + (1 - label) * log(1 - score)
      sl => {
        val score = sl._1
        val label = sl._2
        label * log(score) + (1 - label) * log(1 - score)
      }
    }.reduce(_ + _)
    pLogLoss
  }

  def auc(dataset: DataFrame, labelCol: String, predLabel: String): Double = {
    val scoreAndLabels =
      dataset.select(col(predLabel), col(labelCol).cast(DoubleType)).rdd.map {
        case Row(rawPrediction: Vector, label: Double) => (rawPrediction(1), label)
        case Row(rawPrediction: Double, label: Double) => (rawPrediction, label)
      }

    val ordered = scoreAndLabels.sortBy(x => x._1).zipWithIndex()
    val posRankSum = ordered.map { row => {
      val label = row._1._2
      val order = row._2
      if(label > 0.1)
        order
      else
        0.0
    }}.reduce(_ + _)
    val nRow = dataset.count().toDouble
    val posNum = scoreAndLabels.map {
      sl => sl._2
    }.reduce(_ + _)

    (posRankSum - posNum * (posNum-1) / 2) / (posNum * (nRow - posNum))
  }
}
