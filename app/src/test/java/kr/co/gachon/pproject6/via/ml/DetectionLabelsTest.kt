package kr.co.gachon.pproject6.via.ml

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DetectionLabelsTest {
    @Test
    fun sevenClassSchemaSeparatesPedestrianAndVehicleSignals() {
        assertTrue(DetectionLabels.isPedestrianSignal(DetectionLabels.HUMAN_GREEN))
        assertTrue(DetectionLabels.isPedestrianSignal(DetectionLabels.HUMAN_RED))
        assertFalse(DetectionLabels.isPedestrianSignal(DetectionLabels.VEHICLE_GREEN))
        assertFalse(DetectionLabels.isPedestrianSignal(DetectionLabels.VEHICLE_RED))

        assertTrue(DetectionLabels.isVehicleSignal(DetectionLabels.VEHICLE_GREEN))
        assertTrue(DetectionLabels.isVehicleSignal(DetectionLabels.VEHICLE_RED))
    }

    @Test
    fun activeSchemaPrefersSevenClassModelsWhenPresent() {
        val modelFiles =
            DetectionLabels.modelFilesForActiveSchema(
                listOf(
                    "best_float16_640.tflite",
                    "best_7cls_v2_float16_320.tflite",
                    "best_7cls_v2_float16_416.tflite",
                    "best_7cls_v2_float16_448.tflite",
                    "best_7cls_v2_float16_512.tflite",
                    "best_7cls_v2_float16_640.tflite",
                    "best_7cls_v2_int8_320.tflite",
                    "best_7cls_v2_int8_640.tflite"
                )
            )

        assertEquals(
            listOf(
                "best_7cls_v2_float16_320.tflite",
                "best_7cls_v2_float16_416.tflite",
                "best_7cls_v2_float16_448.tflite",
                "best_7cls_v2_float16_512.tflite",
                "best_7cls_v2_float16_640.tflite",
                "best_7cls_v2_int8_320.tflite",
                "best_7cls_v2_int8_640.tflite"
            ),
            modelFiles
        )
    }
}
