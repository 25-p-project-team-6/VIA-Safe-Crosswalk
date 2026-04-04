package kr.co.gachon.pproject6.via

import org.junit.Assert.assertEquals
import org.junit.Test

class GuidanceRuntimeResetterTest {
    @Test
    fun resetForTrafficLogicDisabledRunsAllResetStepsInOrder() {
        val calls = mutableListOf<String>()
        val resetter = GuidanceRuntimeResetter(
            resetAnalyzer = { calls += "analyzer" },
            resetStabilizer = { calls += "stabilizer" },
            resetCrossingSupport = { calls += "crossingSupport" },
            clearFeedback = { calls += "feedback" },
            afterReset = { calls += "after" }
        )

        resetter.resetForTrafficLogicDisabled()

        assertEquals(
            listOf("analyzer", "stabilizer", "crossingSupport", "feedback", "after"),
            calls
        )
    }

    @Test
    fun resetForPauseMatchesTrafficLogicDisabledResetContract() {
        val calls = mutableListOf<String>()
        val resetter = GuidanceRuntimeResetter(
            resetAnalyzer = { calls += "analyzer" },
            resetStabilizer = { calls += "stabilizer" },
            resetCrossingSupport = { calls += "crossingSupport" },
            clearFeedback = { calls += "feedback" },
            afterReset = { calls += "after" }
        )

        resetter.resetForPause()

        assertEquals(
            listOf("analyzer", "stabilizer", "crossingSupport", "feedback", "after"),
            calls
        )
    }
}
