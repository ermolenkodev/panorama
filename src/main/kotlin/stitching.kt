import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.linalg.inv
import org.jetbrains.kotlinx.multik.jvm.linalg.JvmLinAlg
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.opencv.core.*
import java.util.*
import java.util.stream.IntStream
import kotlin.math.round

/**
 * A class that implements panorama stitching.
 * Standard computer vision pipeline implemented:
 *  - Keypoints detection
 *  - Keypoints matching
 *  - Matches filtering
 *  - Homographies estimation
 *  - Warping images to resulted panorama
 **/
class StitchingAlgorithm(private val homographyAlgorithm: EstimateHomographyAlgorithm) {
    /**
     * Stitch panorama from array of imgs where each image should be stitched to previous
     **/
    fun stitchPanorama(imgs: List<D3Array<Byte>>) : Output<D3Array<Byte>> {
        val n = imgs.size
        return stitchPanorama(imgs, listOf( -1 until  n).flatten())
    }

    /**
     * General case of stitching where images may form a tree.
     * It means that multiple images should be stitched to common parent image
     **/
    fun stitchPanorama(imgs: List<D3Array<Byte>>, parent: List<Int>) : Output<D3Array<Byte>> {
        val result = estimateHomographyForEachImg(imgs, parent)

        val Hs = when (result) {
            is Output.Failure -> return Output.Failure(result.msg, result.e)
            is Output.Success -> result.data
        }

        val bbox = estimateResultBBox(imgs, Hs)

        val resultWidth: Int = bbox.width() + 1
        val resultHeight: Int = bbox.height() + 1

        val panorama = mk.zeros<Byte>(resultHeight, resultWidth, 3)

        val HsInv = try {
            Hs.map { JvmLinAlg.inv(it) }
        } catch (e: RuntimeException) {
            return Output.Failure("Failed to invert one of the Homography matrix", e)
        }

        try {
            warpImages(imgs, bbox, HsInv, panorama)
        } catch (e: RuntimeException) {
            return Output.Failure("Failed to warp imgs to panorama", e)
        }

        return Output.Success(panorama)
    }

    /**
     * Compute homography from each image down to root image of image tree
     **/
    private fun estimateHomographyForEachImg(imgs: List<D3Array<Byte>>,
                                             parent: List<Int>) : Output<List<D2Array<Double>>> {
        val n = imgs.size

        val edges: MutableMap<Pair<Int, Int>, D2Array<Double>> = HashMap()
        for (i in 0 until n) {
            if (parent[i] != -1) {
                val result = homographyAlgorithm.estimateHomography(imgs[i], imgs[parent[i]])

                val H = when (result) {
                    is Output.Failure -> return Output.Failure(
                        "Failed to estimate homography for $i and ${parent[i]} images", result.e
                    )

                    is Output.Success -> result.data
                }

                edges[i to parent[i]] = H
            }
        }

        val Hs: MutableList<D2Array<Double>> = mutableListOf()
        for (i in 0 until n) {
            var p = i
            var H: D2Array<Double> = mk.identity(3)
            while (parent[p] != -1) {
                if (p to parent[p] !in edges) return Output.Failure("Invalid tree of images supplied")

                H = JvmLinAlg.dot(H, edges[p to parent[p]]!!)
                p = parent[p]
            }
            Hs.add(H)
        }

        return Output.Success(Hs)
    }

    private fun estimateResultBBox(imgs: List<D3Array<Byte>>, Hs: List<D2Array<Double>>) : Bbox {
        val bbox = Bbox()
        for (i in imgs.indices) {
            val (h, w, _) = imgs[i].shape
            bbox.grow(pt(0, 0) transform Hs[i])
            bbox.grow(pt(w, 0) transform Hs[i])
            bbox.grow(pt(w, h) transform Hs[i])
            bbox.grow(pt(0, h) transform Hs[i])
        }

        return bbox
    }

    /**
     * Transform every pixel of panorama using computed Homographies.
     * If resulted point correspond one of the imgs copy that pixel value to the panorama
     **/
    private fun warpImages(imgs: List<D3Array<Byte>>, bbox: Bbox,
                           HsInv: List<D2Array<Double>>,
                           result: D3Array<Byte>) {
        val resultWidth: Int = bbox.width() + 1
        val resultHeight: Int = bbox.height() + 1

        /**
         * Parallelization by panorama rows
         * TODO find more efficient way to parallelize this task
         */
        IntStream.range(0, resultHeight)
            .parallel()
            .forEach { y ->
                run {
                    for (x in 0 until resultWidth) {
                        val ptDst = pt(x, y)

                        for (i in imgs.indices) {
                            val ptSrc: Point = (ptDst + bbox.min()) transform HsInv[i]

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
            }
    }
}
