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

import ml.dmlc.xgboost4j.java.XGBoostError
import ml.dmlc.xgboost4j.scala.{DMatrix, EvalTrait}
import org.apache.commons.logging.{Log, LogFactory}

import scala.math._

class Calibration extends EvalTrait {
  private val logger: Log = LogFactory.getLog(classOf[Calibration])
  /**
   * get evaluate metric
   *
   * @return evalMetric
   */
  override def getMetric: String = {
    "calibration"
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
    var posSum = 0.0f
    var probSum = 0.0f
    for (i <- 0 until nrow) {
      posSum += labels(i)
      probSum += predicts(i)(0)
    }
    probSum / posSum
  }
}
