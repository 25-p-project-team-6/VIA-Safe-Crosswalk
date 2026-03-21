package kr.co.gachon.pproject6.via.onboarding

import kr.co.gachon.pproject6.via.ml.InferenceModelProfile
import kr.co.gachon.pproject6.via.ml.ModelQuantization

data class CalibrationProfileResult(
    val profile: InferenceModelProfile,
    val backendLabel: String,
    val averageInputFps: Double,
    val averageDetectFps: Double,
    val averageDetectLatencyMs: Long,
    val averageTotalLatencyMs: Long,
    val compatibilityReportedSupported: Boolean,
    val isUsable: Boolean = true
) {
    fun meetsTarget(targetFps: Double = 15.0): Boolean = isUsable && averageDetectFps >= targetFps
}

object CalibrationSelector {
    const val TARGET_FPS = 15.0

    fun calibrationCandidates(modelFiles: List<String>): List<InferenceModelProfile> {
        return modelFiles
            .filter { it.endsWith(".tflite", ignoreCase = true) }
            .map(InferenceModelProfile::fromFileName)
            .sortedWith(
                compareBy<InferenceModelProfile>(
                    { benchmarkRank(it) },
                    { -(it.inputSize ?: 0) },
                    { it.fileName }
                )
            )
    }

    fun chooseBest(results: List<CalibrationProfileResult>, targetFps: Double = TARGET_FPS): CalibrationProfileResult? {
        if (results.isEmpty()) return null

        val passing = results
            .filter { it.meetsTarget(targetFps) }
            .sortedWith(
                compareByDescending<CalibrationProfileResult> { it.profile.inputSize ?: 0 }
                    .thenBy { benchmarkRank(it.profile) }
                    .thenByDescending { it.averageDetectFps }
            )

        if (passing.isNotEmpty()) {
            return passing.first()
        }

        return results
            .filter { it.isUsable }
            .maxWithOrNull(
                compareBy<CalibrationProfileResult> { it.averageDetectFps }
                    .thenBy { it.profile.inputSize ?: 0 }
            )
            ?: results.first()
    }

    private fun benchmarkRank(profile: InferenceModelProfile): Int {
        return when (profile.quantization) {
            ModelQuantization.FLOAT16 -> 0
            ModelQuantization.FLOAT32 -> 1
            ModelQuantization.INT8 -> 2
            ModelQuantization.UNKNOWN -> 3
        }
    }
}
