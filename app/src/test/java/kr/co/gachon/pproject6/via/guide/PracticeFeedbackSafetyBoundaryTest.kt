package kr.co.gachon.pproject6.via.guide

import java.io.File
import org.junit.Assert.assertFalse
import org.junit.Test

class PracticeFeedbackSafetyBoundaryTest {
    @Test
    fun practicePlayerDoesNotImportLiveGuidanceOrEmergencyActions() {
        val source = sourceFile("guide/PracticeFeedbackPlayer.kt").readText()
        val forbiddenTokens =
            listOf(
                "AdvisoryAssessment",
                "onAdvisoryChanged",
                "Intent",
                "EmergencyContactActivity",
                "Location",
                "Camera"
            )

        forbiddenTokens.forEach { token ->
            assertFalse("practice player must not reference $token", source.contains(token))
        }
    }

    @Test
    fun usageGuidePracticeButtonDoesNotWireRealActions() {
        val source = sourceFile("guide/UsageGuideActivity.kt").readText()
        val forbiddenTokens =
            listOf(
                "onAdvisoryChanged",
                "startActivity",
                "sendTextMessage",
                "requestLocationUpdates",
                "bindToLifecycle"
            )

        forbiddenTokens.forEach { token ->
            assertFalse("usage guide practice UI must not wire $token", source.contains(token))
        }
    }

    private fun sourceFile(relativePath: String): File {
        val fromRoot = File("app/src/main/java/kr/co/gachon/pproject6/via/$relativePath")
        if (fromRoot.exists()) {
            return fromRoot
        }
        return File("src/main/java/kr/co/gachon/pproject6/via/$relativePath")
    }
}
