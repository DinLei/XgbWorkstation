package rank.xgb4j.util;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.time.ZoneId;
import java.util.*;

public class DateUtil {

    private final static SimpleDateFormat SDF;
    static {
        SDF = new SimpleDateFormat("yyyy-MM-dd");
        SDF.setTimeZone(TimeZone.getTimeZone(ZoneId.of("GMT+8")));
    }

    public static int ONE_DAY = 60 * 60 * 24;

    public static List<String> getDateSeq(
            String beginDate, String endDate) throws ParseException {

        Date dBegin = SDF.parse(beginDate);
        Date dEnd = SDF.parse(endDate);

        List<String> lDate = new ArrayList<>();

        lDate.add(SDF.format(dBegin));

        Calendar calBegin = Calendar.getInstance(TimeZone.getTimeZone(ZoneId.of("GMT+8")));
        calBegin.setTime(dBegin);
        Calendar calEnd = Calendar.getInstance(TimeZone.getTimeZone(ZoneId.of("GMT+8")));
        calEnd.setTime(dEnd);

        while (dEnd.after(calBegin.getTime())) {
            calBegin.add(Calendar.DATE, 1);
            lDate.add(SDF.format(calBegin.getTime()));
        }
        return lDate;
    }

    private static String delta(String date, int deltaDay) throws ParseException {
        Date curDate = SDF.parse(date);
        Calendar calendar = Calendar.getInstance(TimeZone.getTimeZone(ZoneId.of("GMT+8")));
        calendar.setTime(curDate);
        calendar.add(Calendar.DAY_OF_YEAR, deltaDay);
        return SDF.format(calendar.getTime());
    }

    public static String before(String date, int deltaDay) throws ParseException {
        return delta(date, -deltaDay);
    }

    public static String after(String date, int deltaDay) throws ParseException {
        return delta(date, deltaDay);
    }

    public static int toTimeStamp(String date) throws ParseException {
        return (int)(SDF.parse(date).getTime() / 1000);
    }

    public static int timeStamp() {
        return (int)(System.currentTimeMillis() / 1000);
    }

    public static Pair<Integer, Integer> getStartEndTimeStampOf(String date) throws ParseException  {
        int start = toTimeStamp(date);
        return new Pair<>(start, start + ONE_DAY);
    }


    public static void main(String[] args) throws ParseException {
        System.out.println(toTimeStamp("2018-08-20"));
        System.out.println(delta("2018-08-20", -10));
        System.out.println(delta("2018-08-20", 12));
        getDateSeq("2018-08-20", "2018-08-31").forEach(System.out::println);
    }
}
