package kr.co.gachon.pproject6.via.safety

import org.junit.Assert.assertTrue
import org.junit.Test

class EmergencyMessageBuilderTest {
    @Test
    fun includesGoogleMapsLinkWhenLocationExists() {
        val message = EmergencyMessageBuilder.build(
            EmergencyLocation(latitude = 37.20313, longitude = 127.114663)
        )

        assertTrue(message.contains("현재 도움이 필요합니다."))
        assertTrue(message.contains("https://maps.google.com/?q=37.203130,127.114663"))
    }

    @Test
    fun fallsBackWhenLocationIsUnavailable() {
        val message = EmergencyMessageBuilder.build(location = null)

        assertTrue(message.contains("현재 위치 확인 불가"))
    }
}
