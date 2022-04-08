import org.jetbrains.kotlinx.multik.api.d2arrayIndices
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.ones
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import org.opencv.core.CvType.CV_8UC3
import org.opencv.core.Mat
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import java.util.DoubleSummaryStatistics

fun Point.asMk() : D1Array<Double> {
    return mk.ndarray(mk[this.x, this.y])
}

fun D1Array<Double>.asCvPoint() : Point {
    return Point(this[0], this[1])
}
//
//fun Mat.asMk() : D2Array<Double> {
//    return mk.d2arrayIndices(3, 3) { i, j -> this.get(i, j) }
//}

fun MatOfKeyPoint.asMatOfPoint() : MatOfPoint2f {
    val points = this.toList().map { it.pt }

    val result = MatOfPoint2f()
    result.fromList(points)

    return result
}

operator fun Point.plus(point: Point) : Point {
    return Point(this.x + point.x, this.y + point.y)
}

fun DoubleArray.toByteArray(): ByteArray {
    return this.map { it.toInt().toByte() }.toByteArray()
}

//fun D3Array<UInt>.asCvMat() : Mat {
//    val (rows, cols, _) = this.shape
//    val mat = Mat(rows, cols, CV_8UC3)
//
//    for (i in 0 until rows) {
//        for (j in 0 until cols) {
//            mat.put(this[i, j].toList().toUIntArray())
//        }
//    }
//}