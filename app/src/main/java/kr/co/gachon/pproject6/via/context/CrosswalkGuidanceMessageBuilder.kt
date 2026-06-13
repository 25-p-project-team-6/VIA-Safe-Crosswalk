package kr.co.gachon.pproject6.via.context

import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sin

data class CrosswalkGuidanceMessage(
    val title: String,
    val detail: String,
    val speechText: String
)

object CrosswalkGuidanceMessageBuilder {
    private const val UNRELIABLE_LOCATION_ACCURACY_METERS = 35f
    private const val DEFAULT_TITLE = "횡단보도 안내"

    fun build(snapshot: CrossingSupportSnapshot): CrosswalkGuidanceMessage {
        val currentLatitude = snapshot.currentLocationLatitude
        val currentLongitude = snapshot.currentLocationLongitude
        if (currentLatitude == null || currentLongitude == null) {
            return unavailable("위치 확인 필요.")
        }

        val mapSnapshot = snapshot.mapProximitySnapshot
        val matchedLatitude = mapSnapshot.matchedLatitude
        val matchedLongitude = mapSnapshot.matchedLongitude
        if (matchedLatitude == null || matchedLongitude == null) {
            return unavailable("근처 횡단보도 없음.")
        }

        val distanceMeters =
            mapSnapshot.distanceMeters
                ?: haversineDistanceMeters(
                    GeoPoint(currentLatitude, currentLongitude),
                    GeoPoint(matchedLatitude, matchedLongitude)
                )
        val distanceText = formatDistanceMeters(distanceMeters)
        val hasUnreliableLocation =
            snapshot.currentLocationAccuracyMeters != null &&
                snapshot.currentLocationAccuracyMeters > UNRELIABLE_LOCATION_ACCURACY_METERS
        val headingDegrees = snapshot.currentHeadingDegrees.takeUnless { hasUnreliableLocation }

        val detail =
            if (headingDegrees != null) {
                val bearingToCrosswalk =
                    bearingDegrees(
                        from = GeoPoint(currentLatitude, currentLongitude),
                        to = GeoPoint(matchedLatitude, matchedLongitude)
                    )
                val direction = relativeDirectionLabel(
                    targetBearingDegrees = bearingToCrosswalk,
                    headingDegrees = headingDegrees
                )
                "횡단보도 ${distanceText} ${direction}."
            } else {
                val reason =
                    if (hasUnreliableLocation) {
                        "GPS 불안정."
                    } else {
                        "방향 확인 중."
                    }
                "횡단보도 ${distanceText}. $reason"
            }

        return CrosswalkGuidanceMessage(
            title = DEFAULT_TITLE,
            detail = detail,
            speechText = detail
        )
    }

    private fun unavailable(detail: String): CrosswalkGuidanceMessage {
        return CrosswalkGuidanceMessage(
            title = DEFAULT_TITLE,
            detail = detail,
            speechText = detail
        )
    }

    private fun formatDistanceMeters(distanceMeters: Float): String {
        val roundedMeters =
            when {
                distanceMeters < 10f -> distanceMeters.roundToInt()
                distanceMeters < 100f -> (distanceMeters / 5f).roundToInt() * 5
                else -> (distanceMeters / 10f).roundToInt() * 10
            }.coerceAtLeast(1)
        return "${roundedMeters}미터"
    }

    private fun bearingDegrees(from: GeoPoint, to: GeoPoint): Float {
        val fromLatitudeRadians = Math.toRadians(from.latitude)
        val toLatitudeRadians = Math.toRadians(to.latitude)
        val longitudeDeltaRadians = Math.toRadians(to.longitude - from.longitude)
        val y = sin(longitudeDeltaRadians) * cos(toLatitudeRadians)
        val x =
            cos(fromLatitudeRadians) * sin(toLatitudeRadians) -
                sin(fromLatitudeRadians) * cos(toLatitudeRadians) * cos(longitudeDeltaRadians)
        return normalizeBearingDegrees(Math.toDegrees(atan2(y, x)).toFloat())
    }

    private fun relativeDirectionLabel(
        targetBearingDegrees: Float,
        headingDegrees: Float
    ): String {
        val delta =
            ((normalizeBearingDegrees(targetBearingDegrees) -
                normalizeBearingDegrees(headingDegrees) + 540f) % 360f) - 180f
        val absDelta = kotlin.math.abs(delta)
        val side = if (delta >= 0f) "오른쪽" else "왼쪽"
        return when {
            absDelta <= 22.5f -> "전방"
            absDelta <= 67.5f -> "전방 $side"
            absDelta <= 112.5f -> side
            absDelta <= 157.5f -> "후방 $side"
            else -> "뒤쪽"
        }
    }

}
