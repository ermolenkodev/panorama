import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.D3Array
import org.opencv.calib3d.Calib3d
import org.opencv.core.*

sealed interface EstimateHomographyAlgorithm {
    fun estimateHomography(img1: D3Array<Byte>, img2: D3Array<Byte>): D2Array<Double>
}

class OpencvImplementation(private val detector: OpencvSiftDetector, private val matcher: OpencvFlannMatcher) :
    EstimateHomographyAlgorithm {
    override fun estimateHomography(img1: D3Array<Byte>, img2: D3Array<Byte>): D2Array<Double> {
        val (keypoints1, descriptors1) = detector.detectAndCompute(img1.asMat())
        val (keypoints2, descriptors2) = detector.detectAndCompute(img2.asMat())

        val matches: List<DMatch> = matcher.match(descriptors1, descriptors2)
            .ratioTestFilter(ratio = 0.7f)
            .map { it.first }

        val points1: List<Point> = matches.map { keypoints1[it.queryIdx].pt }
        val points2: List<Point> = matches.map { keypoints2[it.trainIdx].pt }

        val H: Mat = Calib3d.findHomography(points1.toMatOfPoints(), points2.toMatOfPoints(), Calib3d.RANSAC)

        return H.asD2DoubleArray()
    }
}

