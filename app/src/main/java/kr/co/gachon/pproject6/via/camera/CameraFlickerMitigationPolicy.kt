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
    private const val DEFAULT_TARGET_INPUT_FPS = 20
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
        targetInputFps: Int = DEFAULT_TARGET_INPUT_FPS
    ): CameraFpsRange? {
        availableRanges
            .firstOrNull { it.lower == targetInputFps && it.upper == targetInputFps }
            ?.let { return it }

        // S25-class Camera2 HALs may expose 15-20 but no exact 20-20. Requesting
        // 15-20 lets AE settle at 15 FPS, which is too low for guidance. Prefer a
        // nearby fixed range such as 24-24, then keep processed inference capped at
        // 20 FPS separately for the thermal/battery budget.
        val nearbyFixedRange =
            availableRanges
                .filter {
                    it.lower == it.upper &&
                        it.upper > targetInputFps &&
                        it.upper <= SAFE_FALLBACK_MAX_INPUT_FPS
                }
                .minWithOrNull(
                    compareBy<CameraFpsRange> { it.upper - targetInputFps }
                        .thenBy { it.upper }
                )

        if (nearbyFixedRange != null) {
            return nearbyFixedRange
        }

        return availableRanges
            .filter { it.lower <= targetInputFps && it.upper <= SAFE_FALLBACK_MAX_INPUT_FPS }
            .sortedWith(
                compareByDescending<CameraFpsRange> { it.upper }
                    .thenByDescending { it.lower }
            )
            .firstOrNull()
    }
}
