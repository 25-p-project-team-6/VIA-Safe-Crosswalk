package kr.co.gachon.pproject6.via.guide

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class UsageGuideContentTest {
    @Test
    fun guidePrioritizesOkoLikeInteractiveFeedbackPractice() {
        assertEquals("사용 안내", UsageGuideContent.screenTitle)
        assertEquals("어떤 피드백을 받나요?", UsageGuideContent.feedbackOverview.title)

        val signalTitles = UsageGuidePracticeContent.signalExamples.map { it.title }
        assertEquals(
            listOf("빨간불 예시", "초록불 예시", "주의 필요 예시", "확인 필요 예시"),
            signalTitles
        )
    }

    @Test
    fun guideKeepsOnlyCompactUserNecessarySections() {
        val titles = UsageGuideContent.sections.map { it.title }

        listOf("어떻게 작동하나요?", "휴대폰은 어떻게 들까요?", "피드백이 없으면?", "빠른 조작", "안전 안내").forEach { title ->
            assertTrue("guide should contain $title", titles.contains(title))
        }

        listOf("VIA가 하는 일", "중요 안전 안내", "권한을 쓰는 이유", "주요 기능", "블루투스 버튼", "상태 문구의 의미", "초기 최적화").forEach { oldTitle ->
            assertFalse("guide should remove old long category $oldTitle", titles.contains(oldTitle))
        }
    }

    @Test
    fun guideIncludesClearSafetyDisclaimer() {
        val content = fullGuideText()

        assertTrue(content.contains("보행 판단을 대신하지 않습니다"))
        assertTrue(content.contains("주변"))
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
    fun controlPracticeExamplesExplainRealUserFacingBehavior() {
        val controlText =
            UsageGuidePracticeContent.controlExamples.joinToString("\n") { example ->
                "${example.title}\n${example.description}\n${example.speechText}"
            }

        listOf("가까운 횡단보도의 거리와 방향", "짧게 누르면 주변 횡단보도 안내", "길게 누르면 비상 문자 5초 유예 화면").forEach { phrase ->
            assertTrue("control practice should explain actual behavior for '$phrase'", controlText.contains(phrase))
        }
        assertTrue("control practice should use user-facing Bluetooth wording", controlText.contains("블루투스 버튼"))
        assertTrue("control practice should not mention implementation key wording", !controlText.contains("Space 키"))
    }

    private fun fullGuideText(): String {
        return buildString {
            appendLine(UsageGuideContent.intro)
            appendLine(UsageGuideContent.feedbackOverview.title)
            appendLine(UsageGuideContent.feedbackOverview.body)
            UsageGuideContent.sections.forEach { section ->
                appendLine(section.title)
                appendLine(section.body)
            }
        }
    }
}
