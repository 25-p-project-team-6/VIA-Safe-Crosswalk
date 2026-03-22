package kr.co.gachon.pproject6.via.ml

import org.junit.Assert.assertTrue
import org.junit.Test

class GuidanceTuningDefaultsTest {
    @Test
    fun debugSummaryContainsKeyThresholds() {
        val summary = GuidanceTuningDefaults.toDebugSummary()

        assertTrue(summary.contains("signal hold=250ms"))
        assertTrue(summary.contains("switch hold=400ms"))
        assertTrue(summary.contains("green keep=2500ms"))
        assertTrue(summary.contains("go/stop frames=2"))
        assertTrue(summary.contains("wait frames=3"))
        assertTrue(summary.contains("ready hold=2500ms"))
        assertTrue(summary.contains("walk unknown=1500ms"))
        assertTrue(summary.contains("walk unknown ctx=3500ms"))
        assertTrue(summary.contains("ctx motion=2500ms"))
        assertTrue(summary.contains("ctx gps=4000ms"))
        assertTrue(summary.contains("risk score≥0.35"))
        assertTrue(summary.contains("band=0.20..0.80"))
        assertTrue(summary.contains("wait=8000ms"))
        assertTrue(summary.contains("action=4000ms"))
    }
}
