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

        assertTrue(message.detail.contains("10미터"))
        assertTrue(message.detail.contains("전방"))
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

        assertTrue(message.detail.contains("오른쪽"))
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

        assertTrue(message.detail.contains("10미터"))
        assertTrue(message.detail.contains("GPS 불안정"))
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

        assertTrue(message.detail.contains("10미터"))
        assertTrue(message.detail.contains("방향 확인 중"))
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

        assertTrue(message.detail.contains("위치 확인 필요"))
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

        assertTrue(message.detail.contains("근처 횡단보도 없음"))
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
