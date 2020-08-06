package rank.xgb4s.demo


import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.StringType


object OneHotExample {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().getOrCreate()

    val df = spark.createDataFrame(Seq(
      ("10", 1.0, Array(1,7,3)),
      ("13", 0.0, Array(2,2,3)),
      ("12", 1.0, Array(3,6,3)),
      ("12", 2.0, Array(4,1,3)),
      ("10", 1.0, Array(5,2,5)),
      ("10", 0.0, Array(1,2,8))
    )).toDF("category1", "category2", "category3")

    val dfc = df.select(
      col("category1") +: col("category2") +:
        (0 until 3).map(i => col("category3").getItem(i).as(s"c_$i").cast(StringType)): _* // 变长参数
    )

    dfc.printSchema()

    val indexer = new StringIndexer().setInputCol("category1").setOutputCol("categoryIndex1").setHandleInvalid("keep") //skip keep error

    val idxModel = indexer.fit(df)

    val indexed = idxModel.transform(df)

    indexed.show()

    val encoder1 = new OneHotEncoderEstimator().setInputCols(Array("categoryIndex1", "category2")).setOutputCols(Array("categoryVec1", "categoryVec2"))

    val encoder2 = new OneHotEncoderEstimator().setInputCols(Array("categoryIndex1", "category2")).setOutputCols(Array("categoryVec1", "categoryVec2")) .setDropLast(false)

    val model1 = encoder1.fit(indexed)
    val encoded1 = model1.transform(indexed)
    encoded1.show()

    val model2 = encoder2.fit(indexed)
    val encoded2 = model2.transform(indexed)
    encoded2.show()
    encoded2.col("")


  }
}
