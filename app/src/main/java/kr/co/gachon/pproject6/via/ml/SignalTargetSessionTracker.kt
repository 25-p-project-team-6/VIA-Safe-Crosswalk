package kr.co.gachon.pproject6.via.ml

class SignalTargetSessionTracker(
    private val reacquiredTargetIouThreshold: Float = 0.2f,
    private val minMissingFramesBeforeReset: Int = 5
) {
    private var lastVisibleTarget: NormalizedTargetBox? = null
    private var missingFrameCount: Int = 0

    fun onFrame(target: NormalizedTargetBox?): Boolean {
        if (target == null) {
            if (lastVisibleTarget != null && missingFrameCount < Int.MAX_VALUE) {
                missingFrameCount++
            }
            return false
        }

        val shouldReset =
            lastVisibleTarget != null &&
                missingFrameCount >= minMissingFramesBeforeReset &&
                lastVisibleTarget!!.iou(target) < reacquiredTargetIouThreshold

        lastVisibleTarget = target
        missingFrameCount = 0
        return shouldReset
    }

    fun reset() {
        lastVisibleTarget = null
        missingFrameCount = 0
    }
}

data class NormalizedTargetBox(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float
) {
    fun iou(other: NormalizedTargetBox): Float {
        val intersectionLeft = maxOf(left, other.left)
        val intersectionTop = maxOf(top, other.top)
        val intersectionRight = minOf(right, other.right)
        val intersectionBottom = minOf(bottom, other.bottom)

        if (intersectionLeft >= intersectionRight || intersectionTop >= intersectionBottom) {
            return 0f
        }

        val intersectionArea =
            (intersectionRight - intersectionLeft) * (intersectionBottom - intersectionTop)
        val thisArea = area()
        val otherArea = other.area()
        val unionArea = thisArea + otherArea - intersectionArea

        return if (unionArea > 0f) intersectionArea / unionArea else 0f
    }

    private fun area(): Float {
        return (right - left).coerceAtLeast(0f) * (bottom - top).coerceAtLeast(0f)
    }
}
