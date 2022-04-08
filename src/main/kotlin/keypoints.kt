import org.opencv.core.*
import org.opencv.features2d.FlannBasedMatcher
import org.opencv.features2d.SIFT

class OpencvSiftDetector(private val detector: SIFT) {
    fun detectAndCompute(img: Mat): Pair<List<KeyPoint>, Mat> {
        val keypoints = MatOfKeyPoint()
        val descriptors = Mat()
        detector.detectAndCompute(img, Mat(), keypoints, descriptors)

        return keypoints.toList() to descriptors
    }
}

class OpencvFlannMatcher(private val matcher: FlannBasedMatcher) {
    fun match(descriptors1: Mat, descriptors2: Mat) : List<Pair<DMatch, DMatch>> {
        val matches = mutableListOf<MatOfDMatch>()
        matcher.knnMatch(descriptors1, descriptors2, matches, 2)

        return matches.map { it.toList() }.map { it[0] to it[1] }
    }
}

fun List<Pair<DMatch, DMatch>>.ratioTestFilter(ratio: Float = 0.7f): List<Pair<DMatch, DMatch>> {
    return this.filter { it.first.distance / it.second.distance < ratio }
}
