package kr.co.gachon.pproject6.via.guide

import org.junit.Assert.assertTrue
import org.junit.Assert.assertEquals
import org.junit.Test

class UsageGuideContentTest {
    @Test
    fun guideExplainsRolePermissionsAndMajorFeatures() {
        val content = fullGuideText()

        listOf("보조", "카메라", "위치", "SMS", "신호", "횡단보도", "비상 문자", "블루투스").forEach { keyword ->
            assertTrue("guide should contain $keyword", content.contains(keyword))
        }
    }

    @Test
    fun guideIncludesClearSafetyDisclaimer() {
        val content = fullGuideText()

        assertTrue(content.contains("최종 판단을 대신하지 않습니다"))
        assertTrue(content.contains("주변 상황"))
    }

    @Test
    fun guideDoesNotSayCrossingIsGuaranteedSafe() {
        val content = fullGuideText()
        val forbiddenPhrases = listOf("건너세요", "건너도 된다", "안전하게 건너", "무조건 건너")

        forbiddenPhrases.forEach { phrase ->
            assertTrue("guide should not contain '$phrase'", !content.contains(phrase))
        }
    }

    @Test
    fun guideExplainsFinalAdvisoryStateLabels() {
        val content = fullGuideText()

        listOf(
            "보행자 신호 초록으로 보임",
            "보행자 신호 빨간색으로 보임",
            "초록으로 보이나 주의 필요",
            "신호 확인 불확실",
            "다음 신호 대기 권장"
        ).forEach { label ->
            assertTrue("guide should explain '$label'", content.contains(label))
        }
    }

    @Test
    fun practiceModeIncludesApprovedSignalAndControlExamples() {
        val titles = UsageGuidePracticeContent.allExamples.map { it.title }

        listOf(
            "빨간불 예시 듣기",
            "초록불 예시 듣기",
            "초록불 주의 예시 듣기",
            "확인 필요 예시 듣기",
            "주변 횡단보도 안내",
            "블루투스 짧게 누르기",
            "블루투스 길게 누르기"
        ).forEach { title ->
            assertTrue("practice examples should contain $title", titles.contains(title))
        }
    }

    @Test
    fun practiceExamplesAreClearlySimulationOnly() {
        UsageGuidePracticeContent.allExamples.forEach { example ->
            assertTrue("${example.id} must be simulation only", example.simulationOnly)
            assertTrue("${example.id} speech should be prefixed as practice", example.speechText.startsWith("연습 예시입니다."))
        }
    }

    @Test
    fun signalPracticeExamplesHaveSharedHapticPatterns() {
        assertEquals(4, UsageGuidePracticeContent.signalExamples.size)
        UsageGuidePracticeContent.signalExamples.forEach { example ->
            assertTrue("${example.id} should have haptic pattern", example.hapticPattern != null)
        }
    }

    @Test
    fun controlPracticeExamplesDoNotClaimRealActionsRun() {
        val controlText =
            UsageGuidePracticeContent.controlExamples.joinToString("\n") { example ->
                "${example.title}\n${example.description}\n${example.speechText}"
            }

        listOf("실제 위치를 조회하지 않습니다", "실제 안내를 실행하지 않습니다", "비상 연락을 실행하지 않습니다").forEach { phrase ->
            assertTrue("control practice should state no real action for '$phrase'", controlText.contains(phrase))
        }
    }

    private fun fullGuideText(): String {
        return UsageGuideContent.sections.joinToString("\n") { section ->
            "${section.title}\n${section.body}"
        }
    }
}
