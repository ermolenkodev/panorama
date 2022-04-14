import org.opencv.core.Point
import java.lang.Double.max
import java.lang.Double.min
import kotlin.math.round

class Bbox {
    private val _min: Point = Point(Double.MAX_VALUE, Double.MAX_VALUE)
    private val _max: Point = Point(Double.MIN_VALUE, Double.MIN_VALUE)

    fun grow(pt: Point) {
        _min.x = min(_min.x, pt.x);
        _min.y = min(_min.y, pt.y);

        _max.x = max(_max.x, pt.x);
        _max.y = max(_max.y, pt.y);
    }

    fun width(): Int {
        return round(_max.x - _min.x).toInt()
    }

    fun height(): Int {
        return round(_max.y - _min.y).toInt()
    }

    fun min(): Point {
        return _min
    }
}