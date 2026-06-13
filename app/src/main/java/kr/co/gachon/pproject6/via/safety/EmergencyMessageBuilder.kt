package kr.co.gachon.pproject6.via.safety

import java.util.Locale

object EmergencyMessageBuilder {
    fun build(location: EmergencyLocation?): String {
        val locationText =
            if (location != null) {
                "https://maps.google.com/?q=${formatCoordinate(location.latitude)},${formatCoordinate(location.longitude)}"
            } else {
                "현재 위치 확인 불가"
            }
        return """
            현재 도움이 필요합니다.
            VIA 앱에서 비상 연락 요청이 발생했습니다.
            현재 위치: $locationText
        """.trimIndent()
    }

    private fun formatCoordinate(value: Double): String {
        return String.format(Locale.US, "%.6f", value)
    }
}

data class EmergencyLocation(
    val latitude: Double,
    val longitude: Double
)
