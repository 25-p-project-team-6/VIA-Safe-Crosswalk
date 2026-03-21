package kr.co.gachon.pproject6.via.ml

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class InferenceModelProfileTest {
    @Test
    fun parsesQuantizationAndResolutionHintsFromFilename() {
        val profile = InferenceModelProfile.fromFileName("best_float16_416.tflite")

        assertEquals(ModelQuantization.FLOAT16, profile.quantization)
        assertEquals(416, profile.inputSize)
        assertTrue(profile.recommendedUseGpu)
        assertEquals(640, profile.analysisResolution.width)
        assertEquals(480, profile.analysisResolution.height)
    }

    @Test
    fun int8ProfilesPreferCpuAndSmallerAnalysisResolution() {
        val profile = InferenceModelProfile.fromFileName("best_int8_320.tflite")

        assertEquals(ModelQuantization.INT8, profile.quantization)
        assertFalse(profile.recommendedUseGpu)
        assertEquals(480, profile.analysisResolution.width)
        assertEquals(360, profile.analysisResolution.height)
    }

    @Test
    fun recommendPrefersBalancedGpuProfileWhenGpuIsAvailable() {
        val profile = InferenceModelProfile.recommend(
            modelFiles = listOf(
                "best_float16_640.tflite",
                "best_float16_448.tflite",
                "best_float16_416.tflite",
                "best_int8_320.tflite"
            ),
            gpuSupported = true
        )

        assertNotNull(profile)
        assertEquals("best_float16_640.tflite", profile?.fileName)
    }

    @Test
    fun recommendStillStartsFromFloatProfileWhenGpuCompatibilityIsUnavailable() {
        val profile = InferenceModelProfile.recommend(
            modelFiles = listOf(
                "best_float16_640.tflite",
                "best_float16_416.tflite",
                "best_int8_320.tflite",
                "best_float32_448.tflite"
            ),
            gpuSupported = false
        )

        assertNotNull(profile)
        assertEquals("best_float16_640.tflite", profile?.fileName)
    }
}
