package rank.xgb4s.util


import scala.io.Source
import java.io.{File, FileWriter}

import org.apache.spark.sql.DataFrame

import scala.collection.mutable

object XgbUtils {
  /*
   * Format of featmap.txt: <featureid> <featurename> <q or i or int> :
   * Feature id must be from 0 to number of features, in sorted order.
   * i means this feature is binary indicator feature
   * q means this feature is a quantitative value, such as age, time, can be missing int means this feature is integer value
   *  (when int is hinted, the decision boundary will be integer)
   *
   */
  def featMapConstruct(xFeaList: String, fMapFile: String): Unit = {
    val checkIn = new File(xFeaList)
    if (!checkIn.exists())
      return
    val checkOut = new File(fMapFile)
    if (checkOut.exists())
      checkOut.delete()

    val fin = Source.fromFile(xFeaList)
    val out = new FileWriter(fMapFile,true)
    var counter = 0
    for(line <- fin.getLines()) {
      val newLine = line.stripMargin
      if(newLine != "" && !newLine.startsWith("#")) {
        // name=praiserate;class=S_direct;slot=101;depend=praiserate;is_hash=false;feat_type=0
        // name=nativeflag;class=S_onehot;slot=211;depend=nativeflag;is_hash=false;feat_values=0,1,2
        val tokens = newLine.split(";")
        if(tokens.length >= 3) {
          val fname = tokens(0).split("=")(1)
          var ftype = "i"
          var fValues = ""
          for(i <- 1 until tokens.length) {
            val subToken = tokens(i).split("=")
            if(subToken(0) == "feat_type") {
              if(subToken(1) == "0")
                ftype = "q"
            }else if(subToken(0) == "feat_values") {
              fValues = subToken(1)
            }
          }
          if(fValues == "") {
            out.write(s"$counter\t$fname\t$ftype\n")
            counter += 1
          } else {
            val fSet = fValues.split(",")
            for(cname <- fSet) {
              out.write(s"$counter\t$fname=$cname\t$ftype\n")
              counter += 1
            }
          }
        }
      }
    }
    fin.close(); out.close()
  }


  def paramsParse(params_str: String): Map[String, Any] = {
    var params_map = new mutable.HashMap[String, Any]
    val params_kvs = params_str.stripMargin.split(",")
    val float_pas = Set("subsample", "colsample_bytree", "min_child_weight", "scale_pos_weight", "eta")
    for(pa <- params_kvs) {
      val tokens = pa.stripMargin.split(":")
      if(float_pas.contains(tokens(0)))
        params_map += tokens(0) -> tokens(1).toFloat
      else
        params_map += tokens(0) -> tokens(1).toInt
    }
    params_map.toMap
  }


  def main(args: Array[String]): Unit = {
    featMapConstruct(
      "E:\\coding\\java\\CtrOnSpark\\src\\main\\resources\\dinglei_features_list.conf",
      "E:\\coding\\java\\CtrOnSpark\\src\\main\\resources\\featmap.conf")
  }
}
