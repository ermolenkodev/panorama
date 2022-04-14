import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.jvm.linalg.JvmLinAlg
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.stack
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point

/**
 * Useful extension function for OpenCV Point class
 **/

operator fun Point.plus(point: Point): Point {
    return Point(this.x + point.x, this.y + point.y)
}

fun List<Point>.toMatOfPoints(): MatOfPoint2f {
    val m = MatOfPoint2f()
    m.fromList(this)

    return m
}

infix fun Point.transform(H: D2Array<Double>): Point {
    val transformed = JvmLinAlg.dot(H, mk.ndarray(mk[x, y, 1.0], 3, 1))
    val w = transformed[2, 0]
    if (w == 0.0) throw RuntimeException("Invalid result of homogenous transform")
    return Point(transformed[0, 0], transformed[1, 0])
}

fun pt(x: Int, y: Int): Point {
    return Point(x.toDouble(), y.toDouble())
}

fun pt(x: Double, y: Double): Point {
    return Point(x, y)
}

fun Point.repeatAsColumn(n: Int): D2Array<Double> {
    return mk.stack(List(n) { mk.ndarray(mk[this.x, this.y, 0.0]) }).transpose()
}

fun Point.asMk(): D2Array<Double> {
    return mk.ndarray(mk[this.x, this.y]).reshape(2, 1)
}
