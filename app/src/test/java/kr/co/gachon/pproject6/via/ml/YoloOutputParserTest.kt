package kr.co.gachon.pproject6.via.ml

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class YoloOutputParserTest {
    @Test
    fun layoutNameDescribesYolo26nBatchedNmsOutput() {
        assertEquals(
            "batched_nms_xyxy_score_class",
            YoloOutputParser.layoutName(
                outputCols = 6,
                labels = DetectionLabels.sevenClassLabels
            )
        )
    }

    @Test
    fun layoutNameKeepsSixColumnTwoClassOutputAsClassScore() {
        assertEquals(
            "class_score_cxcywh",
            YoloOutputParser.layoutName(
                outputCols = 6,
                labels = listOf(DetectionLabels.HUMAN_RED, DetectionLabels.HUMAN_GREEN)
            )
        )
    }

    @Test
    fun yolo26nBatchedNmsLayoutUsesScoreAndClassIdColumns() {
        val detections =
            YoloOutputParser.parse(
                output = floatArrayOf(
                    10f, 20f, 110f, 220f, 0.87f, 4f,
                    0f, 0f, 0f, 0f, 0.0f, 0f
                ),
                outputRows = 2,
                outputCols = 6,
                outputIsTransposed = true,
                inputImageWidth = 640,
                inputImageHeight = 640,
                labels = DetectionLabels.sevenClassLabels,
                threshold = 0.15f
            )

        assertEquals(1, detections.size)
        val detection = detections.first()
        assertEquals(DetectionLabels.HUMAN_RED, detection.clsName)
        assertEquals(0.87f, detection.score, 0.0001f)
        assertEquals(10f / 640f, detection.left, 0.0001f)
        assertEquals(20f / 640f, detection.top, 0.0001f)
        assertEquals(110f / 640f, detection.right, 0.0001f)
        assertEquals(220f / 640f, detection.bottom, 0.0001f)
    }

    @Test
    fun yolo26nBatchedNmsLayoutDoesNotTreatClassIdAsConfidence() {
        val detections =
            YoloOutputParser.parse(
                output = floatArrayOf(10f, 20f, 110f, 220f, 0.05f, 6f),
                outputRows = 1,
                outputCols = 6,
                outputIsTransposed = true,
                inputImageWidth = 640,
                inputImageHeight = 640,
                labels = DetectionLabels.sevenClassLabels,
                threshold = 0.15f
            )

        assertTrue(detections.isEmpty())
    }

    @Test
    fun classScoreLayoutStillUsesMaxClassScore() {
        val labels = DetectionLabels.sevenClassLabels
        val row = FloatArray(4 + labels.size)
        row[0] = 320f
        row[1] = 320f
        row[2] = 100f
        row[3] = 120f
        row[4 + labels.indexOf(DetectionLabels.VEHICLE_RED)] = 0.81f

        val detections =
            YoloOutputParser.parse(
                output = row,
                outputRows = 1,
                outputCols = row.size,
                outputIsTransposed = true,
                inputImageWidth = 640,
                inputImageHeight = 640,
                labels = labels,
                threshold = 0.15f
            )

        assertEquals(1, detections.size)
        val detection = detections.first()
        assertEquals(DetectionLabels.VEHICLE_RED, detection.clsName)
        assertEquals(0.81f, detection.score, 0.0001f)
        assertEquals(270f / 640f, detection.left, 0.0001f)
        assertEquals(260f / 640f, detection.top, 0.0001f)
        assertEquals(370f / 640f, detection.right, 0.0001f)
        assertEquals(380f / 640f, detection.bottom, 0.0001f)
    }

    @Test
    fun sixColumnTwoClassOutputRemainsClassScoreLayout() {
        val labels = listOf(DetectionLabels.HUMAN_RED, DetectionLabels.HUMAN_GREEN)
        val detections =
            YoloOutputParser.parse(
                output = floatArrayOf(0.5f, 0.5f, 0.25f, 0.25f, 0.1f, 0.9f),
                outputRows = 1,
                outputCols = 6,
                outputIsTransposed = true,
                inputImageWidth = 640,
                inputImageHeight = 640,
                labels = labels,
                threshold = 0.15f
            )

        assertEquals(1, detections.size)
        val detection = detections.first()
        assertEquals(DetectionLabels.HUMAN_GREEN, detection.clsName)
        assertEquals(0.9f, detection.score, 0.0001f)
    }
}
