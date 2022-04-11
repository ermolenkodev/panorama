import org.jetbrains.kotlinx.multik.ndarray.data.D3Array
import org.opencv.core.*
import org.opencv.features2d.FlannBasedMatcher
import org.opencv.features2d.SIFT

typealias PointsCorrespondences = Pair<List<Point>, List<Point>>

/**
 * Interface aim to encapsulate keypoint matching and filtering logic
 * This is one of the key stages in panorama stitching.
 * It's usually consist of following substages:
 *  - Keypoints detection
 *  - Keypoints matching
 *  - Matches filtering
 **/
sealed interface KeypointsMatchingPipeline {
    fun run(img1: D3Array<Byte>, img2: D3Array<Byte>): Output<PointsCorrespondences>
}

class OpencvKeypointsPipeline(
    private val detector: OpencvSiftDetector,
    private val matcher: OpencvFlannMatcher
) : KeypointsMatchingPipeline {
    override fun run(img1: D3Array<Byte>, img2: D3Array<Byte>): Output<PointsCorrespondences> {
        return try {
            val (keypoints1, descriptors1) = detector.detectAndCompute(img1.asMat())
            val (keypoints2, descriptors2) = detector.detectAndCompute(img2.asMat())

            val matches: List<DMatch> = matcher.match(descriptors1, descriptors2)
                .ratioTestFilter(ratio = 0.7f)
                .map { it.first }

            val points1: List<Point> = matches.map { keypoints1[it.queryIdx].pt }
            val points2: List<Point> = matches.map { keypoints2[it.trainIdx].pt }

            Output.Success(points1 to points2)
        } catch (e: RuntimeException) {
            Output.Failure("Failed to find corresponding keypoint on two images", e)
        }
    }

}

class OpencvSiftDetector(private val detector: SIFT) {
    fun detectAndCompute(img: Mat): Pair<List<KeyPoint>, Mat> {
        val keypoints = MatOfKeyPoint()
        val descriptors = Mat()
        detector.detectAndCompute(img, Mat(), keypoints, descriptors)

        return keypoints.toList() to descriptors
    }
}

class OpencvFlannMatcher(private val matcher: FlannBasedMatcher) {
    fun match(descriptors1: Mat, descriptors2: Mat): List<Pair<DMatch, DMatch>> {
        val matches = mutableListOf<MatOfDMatch>()
        matcher.knnMatch(descriptors1, descriptors2, matches, 2)

        return matches.map { it.toList() }.map { it[0] to it[1] }
    }
}

fun List<Pair<DMatch, DMatch>>.ratioTestFilter(ratio: Float = 0.7f): List<Pair<DMatch, DMatch>> {
    return this.filter { it.first.distance / it.second.distance < ratio }
}
