package kr.co.gachon.pproject6.via.ml

data class AnalysisTargetResolution(
    val width: Int,
    val height: Int
)

enum class ModelQuantization {
    FLOAT16,
    FLOAT32,
    INT8,
    UNKNOWN
}

data class InferenceModelProfile(
    val fileName: String,
    val quantization: ModelQuantization,
    val inputSize: Int?,
    val recommendedUseGpu: Boolean,
    val analysisResolution: AnalysisTargetResolution
) {
    companion object {
        fun fromFileName(fileName: String): InferenceModelProfile {
            val normalized = fileName.lowercase()
            val quantization = when {
                "float16" in normalized -> ModelQuantization.FLOAT16
                "float32" in normalized -> ModelQuantization.FLOAT32
                "int8" in normalized -> ModelQuantization.INT8
                else -> ModelQuantization.UNKNOWN
            }
            val inputSize = Regex("""(\d{3,4})""")
                .find(normalized)
                ?.groupValues
                ?.getOrNull(1)
                ?.toIntOrNull()

            return InferenceModelProfile(
                fileName = fileName,
                quantization = quantization,
                inputSize = inputSize,
                recommendedUseGpu = quantization != ModelQuantization.INT8,
                analysisResolution = preferredAnalysisResolution(inputSize)
            )
        }

        fun recommend(
            modelFiles: List<String>,
            gpuSupported: Boolean
        ): InferenceModelProfile? {
            return modelFiles
                .filter { it.endsWith(".tflite", ignoreCase = true) }
                .map(::fromFileName)
                .minWithOrNull(compareBy({ startupPreferenceRank(it, gpuSupported) }, { it.fileName }))
        }

        private fun startupPreferenceRank(
            profile: InferenceModelProfile,
            gpuSupported: Boolean
        ): Int {
            val quantizationRank = when (profile.quantization) {
                ModelQuantization.FLOAT16 -> 0
                ModelQuantization.FLOAT32 -> if (gpuSupported) 1 else 2
                ModelQuantization.INT8 -> if (gpuSupported) 3 else 1
                ModelQuantization.UNKNOWN -> 4
            }

            val targetInput = if (profile.recommendedUseGpu) 640 else 448
            val sizeRank = profile.inputSize?.let { kotlin.math.abs(it - targetInput) } ?: 10_000
            return quantizationRank * 10_000 + sizeRank
        }

        private fun preferredAnalysisResolution(inputSize: Int?): AnalysisTargetResolution {
            return when {
                inputSize == null -> AnalysisTargetResolution(width = 640, height = 480)
                inputSize <= 320 -> AnalysisTargetResolution(width = 480, height = 360)
                inputSize <= 448 -> AnalysisTargetResolution(width = 640, height = 480)
                inputSize <= 512 -> AnalysisTargetResolution(width = 768, height = 576)
                else -> AnalysisTargetResolution(width = 960, height = 720)
            }
        }
    }

    fun summary(delegateLabel: String): String {
        val inputLabel = inputSize?.toString() ?: "?"
        return "$fileName | $quantization | ${inputLabel}px | $delegateLabel | analysis ${analysisResolution.width}x${analysisResolution.height}"
    }

    fun displayName(): String {
        return when (quantization) {
            ModelQuantization.FLOAT16 -> when {
                (inputSize ?: 0) >= 640 -> "최고 성능 모델"
                (inputSize ?: 0) >= 512 -> "균형형 모델"
                else -> "고속 모델"
            }
            ModelQuantization.FLOAT32 -> when {
                (inputSize ?: 0) >= 640 -> "고정밀 모델"
                else -> "고정밀 중간 모델"
            }
            ModelQuantization.INT8 -> when {
                (inputSize ?: 0) >= 640 -> "절전형 모델"
                else -> "저사양용 모델"
            }
            ModelQuantization.UNKNOWN -> "사용자 지정 모델"
        }
    }

    fun displayNameWithSize(): String {
        val inputLabel = inputSize?.let { "${it}px" } ?: "해상도 미상"
        return "${displayName()} · $inputLabel"
    }
}
