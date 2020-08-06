import scala.collection.mutable.ListBuffer

object TestPathParse {

  def main(args: Array[String]): Unit = {
    val path1 = "hdfs://suninghadoop2/user/predict/dinglei/xfea/dt=20200713/instance"

    val path2 = "hdfs://suninghadoop2/user/predict/dinglei/xfea/dt=20200713/instance/part-00[100-199]"

    val path3 = "hdfs://suninghadoop2/user/predict/dinglei/xfea/dt=20200[716]/instance"

    var left = path1.indexOf('[', 0)
    var right = -1
    if(left >= 0) {
      right = path2.indexOf(']', left)
    }
    if(right > left) {
      val f_val = path2.substring(left+1, right)
      println(s"${left}, ${right}, ${f_val}")
      val Array(start, end) = f_val.split("-")
      for(i <- start.toInt until end.toInt+1) {
        println(path2.substring(0, left)+i+path2.substring(right+1))
      }
    }

    val pa1 = pathRegex(path1)
    pa1.foreach(x => println(x))
    println("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    val pa2 = pathRegex(path2)
    pa2.foreach(x => println(x))
    println("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    val pa3 = pathRegex(path3)
    pa3.foreach(x => println(x))
  }

  def pathRegex(path: String): Array[String] = {
    var left = path.indexOf('[', 0)
    var right = -1
    if(left >= 0) {
      right = path.indexOf(']', left)
      if(right > left) {
        val ans:ListBuffer[String] = ListBuffer()
        val f_val = path.substring(left+1, right)
        val Array(start, end) = f_val.split("-")
        for(i <- start.toInt until end.toInt+1) {
          ans.+=(path.substring(0, left)+i+path.substring(right+1))
        }
        return ans.toArray
      }
    }
    Array(path)
  }
}
