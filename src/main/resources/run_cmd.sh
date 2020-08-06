spark-submit \
--class rank.xgb4j.pipeline.XgbLrPipeline \
--master "local[2]" \
./target/CtrOnSpark-1.0-jar-with-dependencies.jar \
file:///Users/dinglei/javaProjects/CtrOnSpark/src/main/resources/demo/data/agaricus.txt.train


spark-submit \
--class rank.xgb4j.training.xgbLrTraining \
--master local[*] \
./target/CtrOnSpark-1.0-jar-with-dependencies.jar \
file:///root/software/CtrOnSpark/src/main/resources/demo/data/xfea2/part-00197