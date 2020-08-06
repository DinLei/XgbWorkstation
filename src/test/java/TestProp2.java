import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class TestProp2 {
    public static void main(String[] args) {
        InputStream modelIn = TestProp2.class.getClassLoader().getResourceAsStream("lr_model.txt");
        if (modelIn != null) {
            BufferedReader br = new BufferedReader(
                    new InputStreamReader(modelIn)
            );
            String line;
            while (true) {
                try {
                    if ((line = br.readLine()) == null)
                        break;
                    // 一次读入一行数据
                    System.out.println(line);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
}
