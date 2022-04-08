import org.jetbrains.kotlinx.multik.ndarray.data.D3Array
import org.opencv.core.KeyPoint
import org.opencv.core.Mat
import org.opencv.core.MatOfKeyPoint
import org.opencv.features2d.SIFT


sealed interface FeatureDetector {
    fun detect(image: Mat) : Collection<KeyPoint>
}


class OpencvSift(val sift: SIFT) : FeatureDetector {
    override fun detect(image: Mat): Collection<KeyPoint> {
        val cvKeypoints = MatOfKeyPoint()
        sift.detect(image, cvKeypoints)

        return cvKeypoints.toList()
    }
}