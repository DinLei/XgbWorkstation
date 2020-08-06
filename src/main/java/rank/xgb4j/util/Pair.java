package rank.xgb4j.util;

import java.io.Serializable;

public class Pair<H, T> implements Serializable {
    public final H head;
    public final T tail;

    public Pair(H head, T tail) {
        this.head = head;
        this.tail = tail;
    }
}
