package kr.co.gachon.pproject6.via.onboarding

import kr.co.gachon.pproject6.via.ml.InferenceModelProfile
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test

class CalibrationSelectorTest {
    @Test
    fun calibrationCandidatesPreferLargeFloatModelsFirst() {
        val candidates = CalibrationSelector.calibrationCandidates(
            listOf(
                "best_yolo26n_7cls_v2_int8_320.tflite",
                "best_yolo26n_7cls_v2_float16_448.tflite",
                "best_float32_640.tflite",
                "best_yolo26n_7cls_v2_float16_640.tflite"
            )
        )

        assertEquals("best_yolo26n_7cls_v2_float16_640.tflite", candidates.first().fileName)
    }

    @Test
    fun chooseBestPicksHighestResolutionMeetingFifteenFpsTarget() {
        val low = result("best_yolo26n_7cls_v2_float16_448.tflite", 31.0)
        val mid = result("best_yolo26n_7cls_v2_float16_512.tflite", 30.1)
        val high = result("best_yolo26n_7cls_v2_float16_640.tflite", 15.0)

        val best = CalibrationSelector.chooseBest(listOf(low, mid, high))

        assertNotNull(best)
        assertEquals("best_yolo26n_7cls_v2_float16_640.tflite", best?.profile?.fileName)
    }

    @Test
    fun chooseBestFallsBackToFastestWhenNothingMeetsTarget() {
        val faster = result("best_yolo26n_7cls_v2_float16_512.tflite", 14.0)
        val slower = result("best_yolo26n_7cls_v2_float16_640.tflite", 13.0)

        val best = CalibrationSelector.chooseBest(listOf(slower, faster))

        assertNotNull(best)
        assertEquals("best_yolo26n_7cls_v2_float16_512.tflite", best?.profile?.fileName)
    }

    private fun result(fileName: String, fps: Double): CalibrationProfileResult {
        return CalibrationProfileResult(
            profile = InferenceModelProfile.fromFileName(fileName),
            backendLabel = "GPU",
            averageInputFps = 15.0,
            averageDetectFps = fps,
            averageDetectLatencyMs = 0L,
            averageTotalLatencyMs = 0L,
            compatibilityReportedSupported = true
        )
    }
}
