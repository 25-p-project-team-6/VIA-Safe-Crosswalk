package kr.co.gachon.pproject6.via.guide

import org.junit.Assert.assertTrue
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
        val forbiddenPhrases = listOf("건너도 된다", "안전하게 건너", "무조건 건너")

        forbiddenPhrases.forEach { phrase ->
            assertTrue("guide should not contain '$phrase'", !content.contains(phrase))
        }
    }

    private fun fullGuideText(): String {
        return UsageGuideContent.sections.joinToString("\n") { section ->
            "${section.title}\n${section.body}"
        }
    }
}
