package kr.co.gachon.pproject6.via.ml

import org.junit.Assert.assertTrue
import org.junit.Test

class GuidanceTuningDefaultsTest {
    @Test
    fun debugSummaryContainsKeyThresholds() {
        val summary = GuidanceTuningDefaults.toDebugSummary()

        assertTrue(summary.contains("signal hold=250ms"))
        assertTrue(summary.contains("switch hold=400ms"))
        assertTrue(summary.contains("red→green=200ms"))
        assertTrue(summary.contains("green→red=400ms"))
        assertTrue(summary.contains("red flicker bridge=150ms"))
        assertTrue(summary.contains("green keep=2500ms"))
        assertTrue(summary.contains("go confirm=250ms"))
        assertTrue(summary.contains("stop confirm=150ms"))
        assertTrue(summary.contains("wait confirm=350ms"))
        assertTrue(summary.contains("caution confirm=400ms"))
        assertTrue(summary.contains("go hold=1200ms"))
        assertTrue(summary.contains("walk unknown=1200ms"))
        assertTrue(summary.contains("walk unknown matched=2200ms"))
        assertTrue(summary.contains("walk unknown moving=3500ms"))
        assertTrue(summary.contains("walk unknown down=4800ms"))
        assertTrue(summary.contains("ctx motion=2500ms"))
        assertTrue(summary.contains("ctx down=-160.0..-90.0raw/900ms"))
        assertTrue(summary.contains("ctx up=90.0..120.0raw/900ms"))
        assertTrue(summary.contains("ctx gps=4000ms"))
        assertTrue(summary.contains("ctx next=6.0m/5000ms"))
        assertTrue(summary.contains("occupancy score≥0.35"))
        assertTrue(summary.contains("band=0.20..0.80"))
        assertTrue(summary.contains("caution=400ms"))
        assertTrue(summary.contains("advisory high≥75"))
        assertTrue(summary.contains("medium≥55"))
        assertTrue(summary.contains("small<0.015"))
        assertTrue(summary.contains("wait=8000ms"))
        assertTrue(summary.contains("action=4000ms"))
    }
}
