import org.jetbrains.kotlinx.multik.ndarray.data.D3Array
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs

interface ImageIo {
    fun imread(path: String): D3Array<Byte>
    fun imwrite(path: String, img: D3Array<Byte>)
}

class OpencvImageIo : ImageIo {
    override fun imread(path: String): D3Array<Byte> {
        val mat: Mat = Imgcodecs.imread(path)
        return mat.asD3ByteArray()
    }

    override fun imwrite(path: String, img: D3Array<Byte>) {
        val ok: Boolean = Imgcodecs.imwrite(path, img.asMat())
        if (!ok) throw RuntimeException("Failed to write image to disk. Please check $path is correct path to write")
    }
}

fun OpencvImageIo.batchRead(paths: Collection<String>): List<D3Array<Byte>> {
    return paths.map { imread(it) }
}
