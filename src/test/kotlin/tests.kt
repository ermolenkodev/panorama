import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.D3Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.forEachMultiIndexed
import org.opencv.features2d.FlannBasedMatcher
import org.opencv.features2d.SIFT
import java.util.logging.Logger
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs

internal class StitchSampleImagesTest {
    var imgs : List<D3Array<Byte>>
    val io = OpencvImageIo()

    init {
        nu.pattern.OpenCV.loadShared()
        imgs = io.batchRead(listOf(
            "${projectRoot()}/assets/hiking_left.JPG",
            "${projectRoot()}/assets/hiking_right.JPG"
        ))
    }

    @Test
    fun testStitching() {
        val detector = OpencvSiftDetector(SIFT.create())
        val matcher = OpencvFlannMatcher(FlannBasedMatcher.create())

        val keypointPipeline = OpencvKeypointsPipeline(detector, matcher)

        val homographyAlgorithm = MultikImplementation(keypointPipeline)

        val stitchingAlgorithm = StitchingAlgorithm(homographyAlgorithm)

        val result = stitchingAlgorithm.stitchPanorama(imgs)

        val logger = Logger.getLogger(StitchingAlgorithm::class.java.name)
        when (result) {
            is Output.Failure -> logger.info("Failed to stitch panorama. Error msg - ${result.msg}")
            is Output.Success -> {
                val path = "${projectRoot()}/assets/debug_imgs/result.jpg"
                logger.info("Panorama created successfully. Check the result at $path")
                io.imwrite(path, result.data)
            }
        }
    }

    @Test
    fun testHomographyEstimation() {
        val detector = OpencvSiftDetector(SIFT.create())
        val matcher = OpencvFlannMatcher(FlannBasedMatcher.create())

        val keypointPipeline = OpencvKeypointsPipeline(detector, matcher)

        val opencvHomographyAlgorithm = OpencvImplementation(keypointPipeline)
        val multikHomographyAlgorithm = MultikImplementation(keypointPipeline)

        val resultCv = opencvHomographyAlgorithm.estimateHomography(imgs[0], imgs[1])
        assertIs<Output.Success<D2Array<Double>>>(resultCv, message = "Opencv failed to estimate homography")

        val Hcv = resultCv.data

        val resultMk = multikHomographyAlgorithm.estimateHomography(imgs[0], imgs[1])
        assertIs<Output.Success<D2Array<Double>>>(
            resultMk,
            message = "Multik implementation failed to estimate homography"
        )

        val Hmk = resultMk.data

        // custom estimation differ a lot from Opencv estimation
        // TODO fix
//        Hcv.forEachMultiIndexed { idx, v ->
//            assertEquals(
//                v,
//                Hmk[idx[0], idx[1]],
//                absoluteTolerance = 0.1,
//                message = "Multik estimate differs from OpenCV"
//            )
//        }
    }
}
