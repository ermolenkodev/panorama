import org.jetbrains.kotlinx.multik.api.identity
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.linalg.inv
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.jvm.linalg.JvmLinAlg
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.opencv.core.*
import org.opencv.features2d.FlannBasedMatcher
import org.opencv.features2d.SIFT
import org.opencv.imgcodecs.Imgcodecs.imread
import org.opencv.imgcodecs.Imgcodecs.imwrite
import java.util.*
import kotlin.math.round

class StitchingAlgorithm(private val homographyAlgorithm: EstimateHomographyAlgorithm) {
    fun stitchPanorama(imgs: List<D3Array<Byte>>) : D3Array<Byte> {
        val n = imgs.size
        return stitchPanorama(imgs, listOf( -1 until  n).flatten())
    }

    fun stitchPanorama(imgs: List<D3Array<Byte>>, parent: List<Int>) : D3Array<Byte> {
        val n = imgs.size

        val edges: MutableMap<Pair<Int, Int>, D2Array<Double>> = HashMap()
        for (i in 0 until n) {
            if (parent[i] != -1) {
                edges[i to parent[i]] = homographyAlgorithm.estimateHomography(imgs[i], imgs[parent[i]])
            }
        }

        val Hs: MutableList<D2Array<Double>> = mutableListOf()
        for (i in 0 until n) {
            var p = i
            var H: D2Array<Double> = mk.identity(3)
            while (parent[p] != -1) {
                if (p to parent[p] !in edges) throw RuntimeException("Invalid tree of images")
                H = JvmLinAlg.dot(H, edges[p to parent[p]]!!)
                p = parent[p]
            }
            Hs.add(H)
        }

        val bbox = Bbox()
        for (i in 0 until n) {
            val (h, w, _) = imgs[i].shape
            bbox.grow(pt(0, 0) transform Hs[i])
            bbox.grow(pt(w, 0) transform Hs[i])
            bbox.grow(pt(w, h) transform Hs[i])
            bbox.grow(pt(0, h) transform Hs[i])
        }

        val resultWidth: Int = bbox.width() + 1
        val resultHeight: Int = bbox.height() + 1

        val result = mk.zeros<Byte>(resultHeight, resultWidth, 3)

        val HsInv: MutableList<D2Array<Double>> = mutableListOf()
        for (i in Hs.indices) {
            HsInv.add(JvmLinAlg.inv(Hs[i]))
        }

        for (y in 0 until resultHeight) {
            for (x in 0 until resultWidth) {
                val ptDst = pt(x, y)

                for (i in 0 until n) {
                    val ptSrc: Point = ptDst + bbox.min() transform HsInv[i]

                    val (rows, cols, _) = imgs[i].shape

                    val xSrc: Int = round(ptSrc.x).toInt()
                    val ySrc: Int = round(ptSrc.y).toInt()

                    if (xSrc in 0 until cols && ySrc in 0 until rows) {
                        result[y, x] = imgs[i][ySrc, xSrc]
                        break;
                    }
                }
            }
        }

        return result
    }
}

fun main() {
    nu.pattern.OpenCV.loadShared()

    val io = OpencvImageIo()

    val imgs = io.batchRead(listOf(
        "${projectRoot()}/assets/hiking_left.JPG",
        "${projectRoot()}/assets/hiking_right.JPG"
    ))

    val detector = OpencvSiftDetector(SIFT.create())
    val matcher = OpencvFlannMatcher(FlannBasedMatcher.create())
    val homographyAlgorithm = OpencvImplementation(detector, matcher)

    val stitchingAlgorithm = StitchingAlgorithm(homographyAlgorithm)

    val panorama = stitchingAlgorithm.stitchPanorama(imgs)

    io.imwrite("${projectRoot()}/assets/debug_imgs/result.jpg", panorama)
}
