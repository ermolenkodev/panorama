import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.D3Array
import org.jetbrains.kotlinx.multik.ndarray.data.MemoryViewByteArray
import org.opencv.core.CvType.CV_8UC3
import org.opencv.core.Mat

/**
 * Useful extension function to simplify types conversion between OpenCV and multik
 **/

fun Mat.asD3ByteArray(): D3Array<Byte> {
    val data = ByteArray((this.total() * this.channels()).toInt())
    this.get(0, 0, data)

    return mk.ndarray(data).reshape(this.rows(), this.cols(), this.channels())
}

fun D3Array<Byte>.asMat(): Mat {
    val m = Mat(this.shape[0], this.shape[1], CV_8UC3)
    when (this.data) {
        is MemoryViewByteArray -> m.put(0, 0, this.data.getByteArray())
        else -> throw UnsupportedOperationException("Conversion to Mat implemented only for MemoryViewByteArray")
    }

    return m
}

fun Mat.asD2DoubleArray(): D2Array<Double> {
    val data = DoubleArray((this.total() * this.channels()).toInt())
    this.get(0, 0, data)

    return mk.ndarray(data).reshape(this.rows(), this.cols())
}
