package kr.co.gachon.pproject6.via.ml

import org.junit.Assert.assertTrue
import org.junit.Test

class GuidanceTuningDefaultsTest {
    @Test
    fun debugSummaryContainsKeyThresholds() {
        val summary = GuidanceTuningDefaults.toDebugSummary()

        assertTrue(summary.contains("risk score≥0.35"))
        assertTrue(summary.contains("band=0.20..0.80"))
        assertTrue(summary.contains("wait=8000ms"))
        assertTrue(summary.contains("action=4000ms"))
    }
}
