import org.opencv.core.Core
import org.opencv.core.Core.NORM_L2
import org.opencv.core.Core.norm
import org.opencv.core.DMatch
import org.opencv.core.Mat
import java.util.Collections.EMPTY_LIST
import java.util.Collections.swap

class BruteForceMatcher {
    private lateinit var trainDescriptors : Mat

    fun train(trainDesc : Mat)  {
        if (trainDesc.rows() < 2) {
            throw RuntimeException("BruteforceMatcher:: train : needed at least 2 train descriptors")
        }

        trainDescriptors = trainDesc
    }

    fun match(query: Mat, k: Int = 2) :  List<List<DMatch>> {
        if (!this::trainDescriptors.isInitialized) {
            throw RuntimeException("BruteforceMatcher:: Match : matcher is not trained")
        }

        if (k != 2) {
            throw RuntimeException("BruteforceMatcher:: Match : only k = 2 supported")
        }

        val ndesc = query.rows()

        val matches : List<MutableList<DMatch>> = MutableList(ndesc) { mutableListOf() }

        val ntrain = trainDescriptors.rows()

        for (qi in 0 until ndesc) {
            val dst : MutableList<DMatch> = matches[qi]
            for (ti in 0 until ntrain) {
                val match = DMatch()
                val diff = Mat()

                Core.subtract(trainDescriptors.row(ti), query.row(qi), diff)
                match.distance = norm(diff, NORM_L2).toFloat()

                match.imgIdx = 0;
                match.queryIdx = qi
                match.trainIdx = ti

                when {
                    dst.isEmpty() -> dst.add(match)
                    dst.size == 1 -> {
                        dst.add(match)
                        if (dst[0].distance > dst[1].distance) swap(dst, 0, 1)
                    }
                    dst.size == 2 -> {
                        when {
                            dst[0].distance > match.distance -> {
                                dst[1] = dst[0]
                                dst[0] = match
                            }
                            dst[1].distance >= match.distance -> dst[1] = match
                        }
                    }
                    else -> throw RuntimeException("BruteforceMatcher:: match : invalid number of matches")
                }

            }
        }

        return matches
    }
}

