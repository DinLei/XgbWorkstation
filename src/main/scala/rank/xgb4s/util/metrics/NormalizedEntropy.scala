/*
 Copyright (c) 2014 by Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */
package rank.xgb4s.util.metrics

import scala.math._
import ml.dmlc.xgboost4j.java.XGBoostError
import ml.dmlc.xgboost4j.scala.{DMatrix, EvalTrait}
import org.apache.commons.logging.{Log, LogFactory}
import rank.xgb4j.util.CustomEval4j

class NormalizedEntropy extends EvalTrait {
  private val logger: Log = LogFactory.getLog(classOf[NormalizedEntropy])
  /**
   * get evaluate metric
   *
   * @return evalMetric
   */
  override def getMetric: String = {
    "normalized_entropy"
  }

  /**
   * evaluate with predicts and data
   *
   * @param predicts predictions as array
   * @param dmat     data matrix to evaluate
   * @return result of the metric
   */
  override def eval(predicts: Array[Array[Float]], dmat: DMatrix): Float = {
    var labels: Array[Float] = null
    try {
      labels = dmat.getLabel
    } catch {
      case ex: XGBoostError =>
        logger.error(ex)
        return -1f
    }
    val nrow: Int = predicts.length
    var posSum = 0.0d
    var probSum = 0.0d
    for (i <- 0 until nrow) {
      posSum += labels(i)
      probSum += (labels(i) * log(predicts(i)(0)) + (1 - labels(i)) * log(1 - predicts(i)(0)))
    }
    val num = -1.0d / nrow * probSum
    val avg_prob = posSum / nrow
    val den = -1.0d * (avg_prob * log(avg_prob) + (1 - avg_prob) * log(1 - avg_prob))
    (num / den).toFloat
  }
}
