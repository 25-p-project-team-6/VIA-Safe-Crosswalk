package kr.co.gachon.pproject6.via.camera

data class CameraFpsRange(
    val lower: Int,
    val upper: Int
)

data class CameraFlickerMitigationSettings(
    val antibandingMode: Int?,
    val targetFpsRange: CameraFpsRange?
) {
    val isEmpty: Boolean
        get() = antibandingMode == null && targetFpsRange == null
}

object CameraFlickerMitigationPolicy {
    private const val DEFAULT_MAX_INPUT_FPS = 20
    private const val SAFE_FALLBACK_MAX_INPUT_FPS = 30

    fun chooseAntibandingMode(
        availableModes: IntArray?,
        preferredMode: Int,
        autoMode: Int,
        offMode: Int
    ): Int? {
        val modes = availableModes ?: return null
        if (modes.isEmpty()) {
            return null
        }

        return when {
            preferredMode in modes -> preferredMode
            autoMode in modes -> autoMode
            else -> modes.firstOrNull { it != offMode }
        }
    }

    fun chooseTargetFpsRange(
        availableRanges: List<CameraFpsRange>,
        maxInputFps: Int = DEFAULT_MAX_INPUT_FPS
    ): CameraFpsRange? {
        val cappedRanges =
            availableRanges
                .filter { it.upper <= maxInputFps }
                .sortedWith(
                    compareByDescending<CameraFpsRange> { it.upper }
                        .thenByDescending { it.lower }
                )

        if (cappedRanges.isNotEmpty()) {
            return cappedRanges.first()
        }

        // Some Camera2 HALs do not expose an exact <=20 FPS range. In that case,
        // prefer a variable low-start range such as 15-30 instead of forcing 30-30,
        // so AE can still settle below 30 when the device supports it.
        return availableRanges
            .filter { it.lower <= maxInputFps && it.upper <= SAFE_FALLBACK_MAX_INPUT_FPS }
            .sortedWith(
                compareBy<CameraFpsRange> { it.upper }
                    .thenBy { it.lower }
            )
            .firstOrNull()
    }
}
