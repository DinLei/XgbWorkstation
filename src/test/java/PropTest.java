import java.util.Properties;

import static rank.xgb4j.util.PropertiesUtil.getProperties;

public class PropTest {

    public static void main(String[] args) {
        String path = "train.properties";
//        String path = "/Users/dinglei/javaProjects/CtrOnSpark/src/main/resources/train.properties";
        Properties prop = getProperties(path);
        prop.forEach((k, v) -> System.out.println("Key : " + k + ", Value : " + v));
    }

}
