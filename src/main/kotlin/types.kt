import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.jvm.linalg.JvmLinAlg
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.toDoubleArray
import org.opencv.core.CvType.CV_64F
import org.opencv.core.CvType.CV_8UC3
import org.opencv.core.Mat
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Point3
import kotlin.concurrent.thread


fun Mat.asD3ByteArray() : D3Array<Byte> {
    val data = ByteArray((this.total() * this.channels()).toInt())
    this.get(0, 0, data)

    return mk.ndarray(data).reshape(this.rows(), this.cols(), this.channels())
}

fun D3Array<Byte>.asMat() : Mat {
    val m = Mat(this.shape[0], this.shape[1], CV_8UC3)
    when (this.data) {
        is MemoryViewByteArray -> m.put(0, 0, this.data.getByteArray())
        else -> throw UnsupportedOperationException("Conversion to Mat implemented only for MemoryViewByteArray")
    }

    return m
}

fun D2Array<Double>.asDoubleMat() : Mat {
    val m = Mat(this.shape[0], this.shape[1], CV_64F)

    for (i in 0 until this.shape[0]) {
        for (j in 0 until this.shape[1]) {
            m.put(i, j, this[i, j])
        }
    }

    return m
}

fun Mat.asD2DoubleArray() : D2Array<Double> {
    val data = DoubleArray((this.total() * this.channels()).toInt())
    this.get(0, 0, data)

    return mk.ndarray(data).reshape(this.rows(), this.cols())
}

operator fun Point.plus(point: Point) : Point {
    return Point(this.x + point.x, this.y + point.y)
}

fun List<Point>.toMatOfPoints() : MatOfPoint2f {
    val m = MatOfPoint2f()
    m.fromList(this)

    return m
}

fun DoubleArray.toByteArray(): ByteArray {
    return this.map { it.toInt().toByte() }.toByteArray()
}

infix fun Point.transform(H: D2Array<Double>) : Point {
    val transformed  = JvmLinAlg.dot(H,  mk.ndarray(mk[x, y, 1.0], 3, 1))
    val w = transformed[2, 0]
    if (w == 0.0) throw RuntimeException("Invalid result of homogenous transform")
    return Point(transformed[0, 0], transformed[1, 0])
}

fun pt(x: Int, y: Int) : Point {
    return Point(x.toDouble(), y.toDouble())
}


