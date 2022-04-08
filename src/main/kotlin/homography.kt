import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.D3Array
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.calib3d.Calib3d.findHomography as cvFindHomography
import org.opencv.features2d.SIFT

sealed interface Homography {
    fun findHomography(pointsLhs: Collection<D1Array<Double>>, pointsRhs: Collection<D1Array<Double>>): D2Array<Double>
    fun transformPoint(pt: D1Array<Double>, mat: D2Array<Double>): D1Array<Float>
}

//class OpencvHomography : Homography {
//    override fun findHomography(
//        pointsLhs: Collection<D1Array<Double>>,
//        pointsRhs: Collection<D1Array<Double>>
//    ): D2Array<Double> {
//        val cvPointsLhs: List<Point> = pointsLhs.map { it.asCvPoint() }.toList()
//        val cvPointsRhs: List<Point> = pointsRhs.map { it.asCvPoint() }.toList()
//
//        val matLhs = MatOfPoint2f()
//        matLhs.fromList(cvPointsLhs)
//
//        val matRhs = MatOfPoint2f()
//        matRhs.fromList(cvPointsRhs)
//
//        val H: Mat = cvFindHomography(matLhs, matRhs)
//
//        return H.asMk()
//    }
//
//    override fun transformPoint(pt: D1Array<Double>, mat: D2Array<Double>): D1Array<Float> {
//        TODO("Not yet implemented")
//    }
//}
