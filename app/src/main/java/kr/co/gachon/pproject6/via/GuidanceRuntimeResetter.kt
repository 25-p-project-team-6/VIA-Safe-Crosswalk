package kr.co.gachon.pproject6.via

class GuidanceRuntimeResetter(
    private val resetAnalyzer: () -> Unit,
    private val resetStabilizer: () -> Unit,
    private val resetCrossingSupport: () -> Unit,
    private val clearFeedback: () -> Unit,
    private val afterReset: () -> Unit = {}
) {
    fun resetForTrafficLogicDisabled() {
        resetAll()
    }

    fun resetForPause() {
        resetAll()
    }

    private fun resetAll() {
        resetAnalyzer()
        resetStabilizer()
        resetCrossingSupport()
        clearFeedback()
        afterReset()
    }
}
