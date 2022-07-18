import org.jetbrains.kotlinx.multik.api.linalg.solve
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.ones
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.D3Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import org.opencv.calib3d.Calib3d
import org.opencv.core.Mat
import org.opencv.core.Point
import java.util.*
import java.util.logging.Logger

/**
 * Interface aim to encapsulate Homography estimation logic
 **/
interface EstimateHomographyAlgorithm {
    fun estimateHomography(img1: D3Array<Byte>, img2: D3Array<Byte>): Output<D2Array<Double>>
}

/**
 * Homography estimation implementation using Multik
 * It's very naive implementation using Ransac and DLT algorithm
 * The only purpose of this is to practice to work with Multik and Kotlin
 **/
class MultikImplementation(private val keypointsPipeline: KeypointsMatchingPipeline) : EstimateHomographyAlgorithm {

    private val logger = Logger.getLogger(MultikImplementation::class.java.name)

    override fun estimateHomography(img1: D3Array<Byte>, img2: D3Array<Byte>): Output<D2Array<Double>> {
        val result = keypointsPipeline.run(img1, img2)

        val (points1, points2) = when (result) {
            is Output.Failure -> return Output.Failure(result.msg, result.e)
            is Output.Success -> result.data
        }

        return estimateHomographyRANSAC(points1, points2)
    }

    private fun estimateHomographyRANSAC(pointsLhs: List<Point>, pointsRhs: List<Point>): Output<D2Array<Double>> {
        if (pointsLhs.size != pointsRhs.size) {
            return Output.Failure("estimateHomographyRANSAC: points_lhs.size() != points_rhs.size()")
        }

        val nMatches = pointsLhs.size
        val nTrials = 100
        val pxErrorThresh = 2.0

        var bestSupport = 0
        var bestH: D2Array<Double> = mk.ones(3, 3)

        // random 4 points to use for H estimation
        // sampled randomly on each iteration of RANSAC
        val sample: MutableList<Int> = mutableListOf()

        for (trial in 0 until nTrials) {
            try {
                randomSample(sample, nMatches)
            } catch (e: RuntimeException) {
                logger.info(e.message)
                continue
            }

            // random pairs in the following form - point on image1 -> corresponding point on image2
            val pointsPairs = sample.map { randomId -> pointsLhs[randomId] to pointsRhs[randomId] }

            val result = estimateHomography4Points(pointsPairs)

            val H = when (result) {
                is Output.Failure -> continue
                is Output.Success -> result.data
            }

            // count points which projection error (in pixels) less than threshold
            val support = pointsLhs.indices.count { idx ->
                val proj = pointsLhs[idx] transform H

                mk.linalg.norm(proj.asMk() - pointsRhs[idx].asMk()) < pxErrorThresh
            }

            if (support > bestSupport) {
                bestSupport = support
                bestH = H
            }

            if (bestSupport == nMatches) break
        }

        return when (bestSupport) {
            0 -> Output.Failure("estimateHomographyRANSAC: failed to estimate homography")
            else -> Output.Success(bestH)
        }
    }

    /**
     * DLT algorithm
     * Solving system of 8 equation formed from 4 point correspondences
     * Solution is 8 elements of homography matrix and the 9th element is known to be 1.0
     **/
    private fun estimateHomography4Points(pointsPairs: List<Pair<Point, Point>>): Output<D2Array<Double>> {
        val A = mk.zeros<Double>(8, 9)
        for (i in pointsPairs.indices) {
            val points = pointsPairs[i].toList()
            val x = points.map { it.x }
            val y = points.map { it.y }
            val w = listOf(1.0, 1.0)

            A[2 * i] = mk.ndarray(
                mk[0.0, 0.0, 0.0, -x[0] * w[1], -y[0] * w[1], -w[0] * w[1], x[0] * y[1], y[0] * y[1], -w[0] * y[1]]
            )
            A[2 * i + 1] = mk.ndarray(
                mk[x[0] * w[1], y[0] * w[1], w[0] * w[1], 0.0, 0.0, 0.0, -x[0] * x[1], -y[0] * x[1], w[0] * x[1]]
            )
        }

        return try {
            val h = mk.linalg.solve(A[0..8, 0..8], A[0..8, 8])

            Output.Success(
                mk.ndarray(DoubleArray(9) { idx ->
                    when {
                        idx < 8 -> h[idx]
                        else -> 1.0
                    }
                }, 3, 3)
            )
        } catch (e: RuntimeException) {
            Output.Failure("Failed to solve DLT system", e)
        }
    }

    private fun randomSample(dst: MutableList<Int>, maxId: Int, sampleSize: Int = 4, seed: Long = 1007) {
        dst.clear()
        val attempts = 1000
        val rand = Random(seed)

        for (i in 0 until sampleSize) {
            for (k in 0 until attempts) {
                val v: Int = rand.nextInt(maxId)
                if (v !in dst) {
                    dst.add(v)
                    break
                }
            }
            if (dst.size < i + 1) {
                throw RuntimeException("Failed to sample ids")
            }
        }
    }
}

/**
 * Reference implementation of homography estimation using OpenCV
 **/
class OpencvImplementation(private val keypointsPipeline: KeypointsMatchingPipeline) :
    EstimateHomographyAlgorithm {
    override fun estimateHomography(img1: D3Array<Byte>, img2: D3Array<Byte>): Output<D2Array<Double>> {
        val result = keypointsPipeline.run(img1, img2)

        val (points1, points2) = when (result) {
            is Output.Failure -> return Output.Failure(result.msg, result.e)
            is Output.Success -> result.data
        }

        return try {
            val H: Mat = Calib3d.findHomography(points1.toMatOfPoints(), points2.toMatOfPoints(), Calib3d.RANSAC)
            Output.Success(H.asD2DoubleArray())
        } catch (e: RuntimeException) {
            Output.Failure("Opencv failed to estimate homography for two images", e)
        }
    }
}
