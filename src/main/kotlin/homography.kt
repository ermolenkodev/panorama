import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.api.linalg.solve
import org.jetbrains.kotlinx.multik.jvm.linalg.JvmLinAlg
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import org.jetbrains.kotlinx.multik.ndarray.operations.stack
import org.opencv.calib3d.Calib3d
import org.opencv.core.*
import java.util.Random
import java.util.logging.Logger

/**
 * Interface aim to encapsulate Homography estimation logic
 **/
sealed interface EstimateHomographyAlgorithm {
    fun estimateHomography(img1: D3Array<Byte>, img2: D3Array<Byte>): Output<D2Array<Double>>
}

/**
 * Homography estimation implementation using Multik
 * It's very naive implementation using Ransac and DLT algorithm
 * The only purpose of this is to practice to work with Multik and Kotlin
 **/
class MultikImplementation(private val keypointsPipeline: KeypointsMatchingPipeline) : EstimateHomographyAlgorithm {
    companion object {
        val LOG = Logger.getLogger(MultikImplementation::class.java.name)
    }
    override fun estimateHomography(img1: D3Array<Byte>, img2: D3Array<Byte>): Output<D2Array<Double>> {
        val result = keypointsPipeline.run(img1, img2)

        val (points1, points2) = when (result) {
            is Output.Failure -> return Output.Failure(result.msg, result.e)
            is Output.Success -> result.data
        }

        return estimateHomographyRANSAC(points1, points2)
    }
    private fun randomSample(dst: MutableList<Int>, maxId: Int, sampleSize: Int, seed: Long = 1007) {
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

    private fun estimateHomographyRANSAC(pointsLhs: List<Point>, pointsRhs: List<Point>) : Output<D2Array<Double>> {
        if (pointsLhs.size != pointsRhs.size) {
            return Output.Failure("estimateHomographyRANSAC: points_lhs.size() != points_rhs.size()")
        }
        
        val nMatches = pointsLhs.size
        val nTrials = 100
        val nSamples = 4
        val seed = 1007L
        val pxErrorThresh = 2.0
        var bestSupport = 0
        var bestH : D2Array<Double> = mk.ones(3, 3)
        
        val sample : MutableList<Int> = mutableListOf()

        for (trial in 0 until  nTrials) {
            try {
                randomSample(sample, nMatches, nSamples, seed)
            } catch (e: RuntimeException) {
                LOG.info(e.message)
                continue
            }

            val result = estimateHomography4Points(
                pointsLhs[sample[0]], pointsLhs[sample[1]], pointsLhs[sample[2]], pointsLhs[sample[3]],
                pointsRhs[sample[0]], pointsRhs[sample[1]], pointsRhs[sample[2]], pointsRhs[sample[3]]
            )

            val H = when (result) {
                is Output.Failure -> continue
                is Output.Success -> result.data
            }

            var support = 0
            for (idx in 0 until nMatches) {
                val proj = pointsLhs[idx] transform H
                if (JvmLinAlg.norm(proj.asMk() - pointsRhs[idx].asMk()) < pxErrorThresh) {
                    ++support
                }
            }

            if (support > bestSupport) {
                bestSupport = support
                bestH = H
                if (bestSupport == nMatches) {
                    break
                }
            }
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
    private fun estimateHomography4Points(
        l0: Point,
        l1: Point,
        l2: Point,
        l3: Point,
        r0: Point,
        r1: Point,
        r2: Point,
        r3: Point,
    ): Output<D2Array<Double>> {
        val xs0 = mk.ndarray(mk[l0.x, l1.x, l2.x, l3.x])
        val xs1 = mk.ndarray(mk[r0.x, r1.x, r2.x, r3.x])
        val ys0 = mk.ndarray(mk[l0.y, l1.y, l2.y, l3.y])
        val ys1 = mk.ndarray(mk[r0.y, r1.y, r2.y, r3.y])
        val ws0 = mk.ndarray(mk[1.0, 1.0, 1.0, 1.0])
        val ws1 = mk.ndarray(mk[1.0, 1.0, 1.0, 1.0])

        val A = mk.zeros<Double>(8, 9)
        for (i in 0 until 4) {
            val (x0, x1, y0) = listOf(xs0[i], xs1[i], ys0[i])
            val (y1, w0, w1) = listOf(ys1[i], ws0[i], ws1[i])
            A[2*i] = mk.ndarray(mk[0.0, 0.0, 0.0, -x0*w1, -y0*w1, -w0*w1, x0*y1, y0*y1, -w0*y1])
            A[2*i+1] = mk.ndarray(mk[x0*w1, y0*w1, w0*w1, 0.0, 0.0, 0.0, -x0*x1, -y0*x1, w0*x1])
        }

        return try {
            val h = JvmLinAlg.solve(A[0..8, 0..8], A[0..8, 8])

            Output.Success(
                mk.ndarray(DoubleArray(9) {
                        idx -> when {
                            idx < 8 -> h[idx]
                            else -> 1.0
                        }
                }).reshape(3, 3)
            )
        } catch (e: RuntimeException) {
            Output.Failure("Failed to solve DLT system", e)
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
