public class MyTest {
    public static void main(String[] args) {
        String s1 = "abc|$|def|$|ghi|jkl";

        String subString = "";
        String finalSubString = "";

        // 由后向前查找字符串位置，找不到返回-1
        int idx = s1.lastIndexOf("|$|");
        if(idx != -1) {
            subString = s1.substring(idx);
            finalSubString = s1.substring(idx+3);
        }

        System.out.println(
                "last_idx: " + idx +
                ", sub_string: " + subString +
                ", final_sub_string: " + finalSubString
        );
    }
}
