package kr.co.gachon.pproject6.via.ml

import android.graphics.RectF
import kr.co.gachon.pproject6.via.ui.OverlayView
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ObjectTrackerScoringTest {
    @Test
    fun largerOffCenterSignalCanBeatSmallerCenteredSignal() {
        val smallerCentered =
            calculateTargetPriority(
                confidence = 0.9f,
                normalizedArea = 0.010f,
                centerDistance = 0.03f
            )
        val largerOffCenter =
            calculateTargetPriority(
                confidence = 0.9f,
                normalizedArea = 0.020f,
                centerDistance = 0.28f
            )

        assertTrue(largerOffCenter > smallerCentered)
    }

    @Test
    fun centerStillWinsWhenSizesAreNearlyEqual() {
        val centered =
            calculateTargetPriority(
                confidence = 0.9f,
                normalizedArea = 0.012f,
                centerDistance = 0.04f
            )
        val slightlyLargerOffCenter =
            calculateTargetPriority(
                confidence = 0.9f,
                normalizedArea = 0.013f,
                centerDistance = 0.30f
            )

        assertTrue(centered > slightlyLargerOffCenter)
    }

    @Test
    fun trackerIgnoresVehicleSignalsWhenNoHumanSignalExists() {
        val tracker = ObjectTracker()

        val target =
            tracker.selectTarget(
                listOf(
                    box(DetectionLabels.VEHICLE_GREEN, score = 0.99f, left = 0.2f),
                    box(DetectionLabels.VEHICLE_RED, score = 0.95f, left = 0.5f)
                )
            )

        assertNull(target)
    }

    private fun box(label: String, score: Float, left: Float): OverlayView.BoundingBox {
        return OverlayView.BoundingBox(
            RectF(left, 0.2f, left + 0.2f, 0.5f),
            label,
            score
        )
    }
}
