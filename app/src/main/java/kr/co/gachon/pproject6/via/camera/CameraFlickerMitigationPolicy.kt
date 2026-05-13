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

    fun chooseTargetFpsRange(availableRanges: List<CameraFpsRange>): CameraFpsRange? {
        return availableRanges
            .filter { it.upper <= 30 }
            .sortedWith(
                compareByDescending<CameraFpsRange> { it.upper }
                    .thenByDescending { it.lower }
            )
            .firstOrNull()
    }
}
