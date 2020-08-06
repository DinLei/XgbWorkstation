import java.util.Properties

object TestProp {

  def main(args: Array[String]): Unit = {

    val prop = new Properties()
    val inputStream = TestProp.getClass.getClassLoader.getResourceAsStream("train.properties")
    prop.load(inputStream)

    println(prop.getProperty("objective"))
    println(prop.getProperty("num_workers", "3"))
  }
}
