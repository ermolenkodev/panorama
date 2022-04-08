import org.opencv.calib3d.Calib3d
import org.opencv.calib3d.Calib3d.RANSAC
import org.opencv.core.*
import org.opencv.core.Core.perspectiveTransform
import org.opencv.core.CvType.CV_64FC1
import org.opencv.core.CvType.CV_8UC3
import org.opencv.features2d.FlannBasedMatcher
import org.opencv.features2d.SIFT
import org.opencv.imgcodecs.Imgcodecs.imread
import org.opencv.imgcodecs.Imgcodecs.imwrite
import java.util.*
import kotlin.math.round


fun transformPointCV(p : Point, m: Mat) : Point {
    val point = MatOfPoint2f()
    point.alloc(1)
    point.put(0, 0, p.x, p.y)

    perspectiveTransform(point, point, m)

    return Point(point.get(0, 0)[0], point.get(0, 0)[1])
}

fun stitchPanorama(imgs: List<Mat>, parent: List<Int>, homographyBuilder: (im1: Mat, im2: Mat) -> Mat) {
    val n = imgs.size

    val edges: MutableMap<Pair<Int, Int>, Mat> = HashMap()
    for (i in 0 until n) {
        if (parent[i] != -1) {
            edges[i to parent[i]] = homographyBuilder(imgs[i], imgs[parent[i]])
        }
    }

    val Hs: MutableList<Mat> = mutableListOf()
    for (i in 0 until n) {
        var p = i
        var H = Mat.eye(3, 3, CV_64FC1)
        while (parent[p] != -1) {
            Core.gemm(H, edges[p to parent[p]], 1.0, Mat(), 0.0, H, 0);
//            H = edges[p to parent[p]]!!.mul(H)
            p = parent[p]
        }
        Hs.add(H)
    }

    val bbox = Bbox()
    for (i in 0 until n) {
        val w = imgs[i].cols().toDouble()
        val h = imgs[i].rows().toDouble()
        bbox.grow(transformPointCV(Point(0.0, 0.0), Hs[i]))
        bbox.grow(transformPointCV(Point(w, 0.0), Hs[i]))
        bbox.grow(transformPointCV(Point(w, h), Hs[i]))
        bbox.grow(transformPointCV(Point(0.0, h), Hs[i]))
    }

    val resultWidth: Int = bbox.width() + 1
    val resultHeight: Int = bbox.height() + 1

    val result: Mat = Mat.zeros(resultHeight, resultWidth, CV_8UC3)

    val HsInv: MutableList<Mat> = mutableListOf()
    for (i in Hs.indices) {
        HsInv.add(Hs[i].inv())
    }

    for (y in 0 until resultHeight) {
        for (x in 0 until resultWidth) {
            val ptDst = Point(x.toDouble(), y.toDouble())
            for (i in 0 until n) {
                val ptSrc: Point = transformPointCV(ptDst + bbox.min(), HsInv[i])

                val xSrc: Int = round(ptSrc.x).toInt()
                val ySrc: Int = round(ptSrc.y).toInt()
                if (xSrc >= 0 && xSrc < imgs[i].cols() && ySrc >= 0 && ySrc < imgs[i].rows()) {
                    val pixel: DoubleArray = imgs[i].get(ySrc, xSrc)
                    result.put(y, x, pixel.toByteArray())
                    break;
                }
            }
        }
    }

    imwrite("result.jpg", result)
}

fun findHomographyCV(img1: Mat, img2: Mat) : Mat {
    val detector: SIFT = SIFT.create()

    val keypoints1 = MatOfKeyPoint()
    val descriptors1 = Mat()
    detector.detectAndCompute(img1, Mat(), keypoints1, descriptors1)

    val keypoints2 = MatOfKeyPoint()
    val descriptors2 = Mat()
    detector.detectAndCompute(img2, Mat(), keypoints2, descriptors2)

    val matcher = FlannBasedMatcher.create()

    val matches = mutableListOf<MatOfDMatch>()
    matcher.knnMatch(descriptors1, descriptors2, matches, 2)
//    val matcher = BruteForceMatcher()
//    matcher.train(descriptors2)
//
//    val matches : List<List<DMatch>> = matcher.match(descriptors1)

    val testRatio = 0.7f

//    val goodMatches : List<DMatch> = matches
//        .toList()
//        .filter { twoBest -> twoBest[0].distance / twoBest[1].distance <= testRatio }
//        .map { it[0] }
//        .toList()

    val goodMatches = mutableListOf<DMatch>()
    val iterator = matches.iterator()
    while (iterator.hasNext()) {
        val matOfDMatch = iterator.next()
        if (matOfDMatch.toArray()[0].distance / matOfDMatch.toArray()[1].distance < testRatio) {
            goodMatches.add(matOfDMatch.toArray()[0])
        }
    }

    val keypointsList1 = keypoints1.toList()
    val keypointsList2 = keypoints2.toList()

    val points1 : MutableList<Point> = mutableListOf()
    val points2 : MutableList<Point> = mutableListOf()
    for (match in goodMatches) {
        points1.add(keypointsList1[match.queryIdx].pt)
        points2.add(keypointsList2[match.trainIdx].pt);
    }

    val pointsMat1 = MatOfPoint2f()
    pointsMat1.fromList(points1)

    val pointsMat2 = MatOfPoint2f()
    pointsMat2.fromList(points2)

    val H: Mat = Calib3d.findHomography(pointsMat1, pointsMat2, RANSAC)
    println(H.dump())

    return H
}




fun main() {
    nu.pattern.OpenCV.loadShared()

    val img1: Mat = imread("${projectRoot()}/assets/hiking_left.JPG")
    val img2: Mat = imread("${projectRoot()}/assets/hiking_right.JPG")

    stitchPanorama(listOf(img1, img2), listOf(-1, 0), ::findHomographyCV)
}
