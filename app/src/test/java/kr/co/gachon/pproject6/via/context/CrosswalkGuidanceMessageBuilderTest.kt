package kr.co.gachon.pproject6.via.context

import org.junit.Assert.assertTrue
import org.junit.Test

class CrosswalkGuidanceMessageBuilderTest {
    @Test
    fun guidanceIncludesDistanceAndForwardDirectionWhenHeadingIsKnown() {
        val message =
            CrosswalkGuidanceMessageBuilder.build(
                CrossingSupportSnapshot(
                    currentLocationLatitude = 37.0,
                    currentLocationLongitude = 127.0,
                    currentLocationAccuracyMeters = 8f,
                    currentHeadingDegrees = 0f,
                    mapProximitySnapshot = MapProximitySnapshot(
                        matchedLatitude = 37.00009,
                        matchedLongitude = 127.0,
                        distanceMeters = 10f
                    )
                )
        )

        assertTrue(message.detail.contains("약 10미터"))
        assertTrue(message.detail.contains("전방에"))
    }

    @Test
    fun guidanceUsesRightSideDirectionRelativeToHeading() {
        val message =
            CrosswalkGuidanceMessageBuilder.build(
                CrossingSupportSnapshot(
                    currentLocationLatitude = 37.0,
                    currentLocationLongitude = 127.0,
                    currentLocationAccuracyMeters = 8f,
                    currentHeadingDegrees = 0f,
                    mapProximitySnapshot = MapProximitySnapshot(
                        matchedLatitude = 37.0,
                        matchedLongitude = 127.0001,
                        distanceMeters = 9f
                    )
                )
            )

        assertTrue(message.detail.contains("오른쪽 방향"))
    }

    @Test
    fun guidanceSuppressesDirectionWhenLocationAccuracyIsLow() {
        val message =
            CrosswalkGuidanceMessageBuilder.build(
                CrossingSupportSnapshot(
                    currentLocationLatitude = 37.0,
                    currentLocationLongitude = 127.0,
                    currentLocationAccuracyMeters = 50f,
                    currentHeadingDegrees = 0f,
                    mapProximitySnapshot = MapProximitySnapshot(
                        matchedLatitude = 37.00009,
                        matchedLongitude = 127.0,
                        distanceMeters = 10f
                    )
                )
            )

        assertTrue(message.detail.contains("약 10미터 거리에 있습니다"))
        assertTrue(message.detail.contains("방향 안내는 생략합니다"))
    }

    @Test
    fun guidanceSuppressesDirectionWhenHeadingIsMissing() {
        val message =
            CrosswalkGuidanceMessageBuilder.build(
                CrossingSupportSnapshot(
                    currentLocationLatitude = 37.0,
                    currentLocationLongitude = 127.0,
                    currentLocationAccuracyMeters = 8f,
                    currentHeadingDegrees = null,
                    mapProximitySnapshot = MapProximitySnapshot(
                        matchedLatitude = 37.00009,
                        matchedLongitude = 127.0,
                        distanceMeters = 10f
                    )
                )
            )

        assertTrue(message.detail.contains("약 10미터 거리에 있습니다"))
        assertTrue(message.detail.contains("이동 방향을 아직 확인하지 못해"))
        assertTrue(message.detail.contains("방향 안내는 생략합니다"))
    }

    @Test
    fun guidanceReportsUnavailableWhenLocationIsMissing() {
        val message =
            CrosswalkGuidanceMessageBuilder.build(
                CrossingSupportSnapshot(
                    currentHeadingDegrees = 0f,
                    mapProximitySnapshot = MapProximitySnapshot(
                        matchedLatitude = 37.00009,
                        matchedLongitude = 127.0,
                        distanceMeters = 10f
                    )
                )
            )

        assertTrue(message.detail.contains("현재 위치를 확인하지 못했습니다"))
    }

    @Test
    fun guidanceReportsUnavailableWhenNoMapMatchExists() {
        val message =
            CrosswalkGuidanceMessageBuilder.build(
                CrossingSupportSnapshot(
                    currentLocationLatitude = 37.0,
                    currentLocationLongitude = 127.0,
                    currentHeadingDegrees = 0f
                )
            )

        assertTrue(message.detail.contains("횡단보도 정보가 없습니다"))
    }

    @Test
    fun guidanceDoesNotClaimSignalStateOrCrossingPermission() {
        val message =
            CrosswalkGuidanceMessageBuilder.build(
                CrossingSupportSnapshot(
                    currentLocationLatitude = 37.0,
                    currentLocationLongitude = 127.0,
                    currentLocationAccuracyMeters = 8f,
                    currentHeadingDegrees = 0f,
                    mapProximitySnapshot = MapProximitySnapshot(
                        matchedLatitude = 37.00009,
                        matchedLongitude = 127.0,
                        distanceMeters = 10f
                    )
                )
            )

        val forbiddenPhrases = listOf("초록", "빨간", "신호", "건너세요", "건너도")
        forbiddenPhrases.forEach { phrase ->
            assertTrue(
                "message should not contain '$phrase': ${message.detail}",
                !message.detail.contains(phrase)
            )
        }
    }
}
