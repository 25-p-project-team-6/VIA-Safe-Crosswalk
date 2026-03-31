package kr.co.gachon.pproject6.via.context

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class MapProximityEngineTest {
    @Test
    fun requiresTwoConsecutiveAcceptedFixesBeforeEnteringNearState() {
        val engine = MapProximityEngine()
        val feature = crosswalkFeature(id = "crosswalk-a", latitude = 37.4500, longitude = 127.1280)
        val point = GeoPoint(latitude = 37.45005, longitude = 127.1280)

        val first = engine.update(point, 5f, null, listOf(feature), "bundled-v1", false)
        val second = engine.update(point, 5f, null, listOf(feature), "bundled-v1", false)

        assertFalse(first.isNearKnownFeature)
        assertTrue(second.isNearKnownFeature)
        assertEquals("crosswalk-a", second.matchedFeatureId)
        assertEquals(37.4500, second.matchedLatitude!!, 0.000001)
        assertEquals(127.1280, second.matchedLongitude!!, 0.000001)
    }

    @Test
    fun preservesActiveMatchUntilExitRadiusIsExceeded() {
        val engine = MapProximityEngine()
        val feature = crosswalkFeature(id = "crosswalk-a", latitude = 37.4500, longitude = 127.1280)
        val enterPoint = GeoPoint(latitude = 37.45005, longitude = 127.1280)
        val stillInsideExit = GeoPoint(latitude = 37.45040, longitude = 127.1280)

        engine.update(enterPoint, 4f, null, listOf(feature), "bundled-v1", false)
        engine.update(enterPoint, 4f, null, listOf(feature), "bundled-v1", false)
        val retained = engine.update(stillInsideExit, 4f, null, listOf(feature), "bundled-v1", false)

        assertTrue(retained.isNearKnownFeature)
        assertEquals("crosswalk-a", retained.matchedFeatureId)
    }

    @Test
    fun switchesToCloserCandidateBeforeOldExitRadiusWhenItWinsByLargeMargin() {
        val engine = MapProximityEngine()
        val first = crosswalkFeature(id = "crosswalk-a", latitude = 37.45000, longitude = 127.12800)
        val second = crosswalkFeature(id = "crosswalk-b", latitude = 37.45054, longitude = 127.12800)
        val initialPoint = GeoPoint(latitude = 37.45003, longitude = 127.12800)
        val movedPoint = GeoPoint(latitude = 37.45049, longitude = 127.12800)

        engine.update(initialPoint, 4f, null, listOf(first, second), "bundled-v1", false)
        engine.update(initialPoint, 4f, null, listOf(first, second), "bundled-v1", false)

        val firstSwitchAttempt = engine.update(movedPoint, 4f, null, listOf(first, second), "bundled-v1", false)
        val confirmedSwitch = engine.update(movedPoint, 4f, null, listOf(first, second), "bundled-v1", false)

        assertEquals("crosswalk-a", firstSwitchAttempt.matchedFeatureId)
        assertEquals("crosswalk-b", confirmedSwitch.matchedFeatureId)
    }

    @Test
    fun doesNotSwitchInsideDenseClusterForSmallDistanceDifference() {
        val engine = MapProximityEngine()
        val first = crosswalkFeature(id = "crosswalk-a", latitude = 37.45000, longitude = 127.12800)
        val second = crosswalkFeature(id = "crosswalk-b", latitude = 37.45008, longitude = 127.12800)
        val initialPoint = GeoPoint(latitude = 37.45001, longitude = 127.12800)
        val slightlyCloserToSecond = GeoPoint(latitude = 37.45005, longitude = 127.12800)

        engine.update(initialPoint, 4f, null, listOf(first, second), "bundled-v1", false)
        engine.update(initialPoint, 4f, null, listOf(first, second), "bundled-v1", false)

        val afterMove = engine.update(slightlyCloserToSecond, 4f, null, listOf(first, second), "bundled-v1", false)
        val secondAfterMove = engine.update(slightlyCloserToSecond, 4f, null, listOf(first, second), "bundled-v1", false)

        assertEquals("crosswalk-a", afterMove.matchedFeatureId)
        assertEquals("crosswalk-a", secondAfterMove.matchedFeatureId)
    }

    @Test
    fun higherAccuracyPenaltyKeepsActiveMatchUntilDifferenceIsMuchLarger() {
        val engine = MapProximityEngine()
        val first = crosswalkFeature(id = "crosswalk-a", latitude = 37.45000, longitude = 127.12800)
        val second = crosswalkFeature(id = "crosswalk-b", latitude = 37.45054, longitude = 127.12800)
        val initialPoint = GeoPoint(latitude = 37.45003, longitude = 127.12800)
        val movedPoint = GeoPoint(latitude = 37.45049, longitude = 127.12800)

        engine.update(initialPoint, 4f, null, listOf(first, second), "bundled-v1", false)
        engine.update(initialPoint, 4f, null, listOf(first, second), "bundled-v1", false)

        val firstSwitchAttempt = engine.update(movedPoint, 20f, null, listOf(first, second), "bundled-v1", false)
        val secondSwitchAttempt = engine.update(movedPoint, 20f, null, listOf(first, second), "bundled-v1", false)

        assertEquals("crosswalk-a", firstSwitchAttempt.matchedFeatureId)
        assertEquals("crosswalk-a", secondSwitchAttempt.matchedFeatureId)
    }

    @Test
    fun poorAccuracyDoesNotCreateNewMatch() {
        val engine = MapProximityEngine()
        val feature = crosswalkFeature(id = "crosswalk-a", latitude = 37.4500, longitude = 127.1280)
        val point = GeoPoint(latitude = 37.45005, longitude = 127.1280)

        val snapshot = engine.update(point, 40f, null, listOf(feature), "bundled-v1", false)

        assertFalse(snapshot.isNearKnownFeature)
        assertEquals("bundled-v1", snapshot.datasetVersion)
    }

    @Test
    fun headingBreaksNearDistanceTiesWhenApproachBearingExists() {
        val engine = MapProximityEngine()
        val northSouth = crosswalkFeature(
            id = "crosswalk-ns",
            latitude = 37.4500,
            longitude = 127.1280,
            approachBearings = listOf(0f, 180f)
        )
        val eastWest = crosswalkFeature(
            id = "crosswalk-ew",
            latitude = 37.45002,
            longitude = 127.1280,
            approachBearings = listOf(90f, 270f)
        )
        val point = GeoPoint(latitude = 37.45008, longitude = 127.1280)

        engine.update(point, 5f, 92f, listOf(northSouth, eastWest), "bundled-v1", false)
        val snapshot =
            engine.update(point, 5f, 92f, listOf(northSouth, eastWest), "bundled-v1", false)

        assertTrue(snapshot.isNearKnownFeature)
        assertEquals("crosswalk-ew", snapshot.matchedFeatureId)
    }

    private fun crosswalkFeature(
        id: String,
        latitude: Double,
        longitude: Double,
        approachBearings: List<Float> = emptyList()
    ): MapFeatureRecord {
        return MapFeatureRecord(
            id = id,
            kind = MapFeatureKind.CROSSWALK,
            point = GeoPoint(latitude = latitude, longitude = longitude),
            triggerRadiusMeters = 35f,
            exitRadiusMeters = 55f,
            approachBearings = approachBearings,
            regionTileId = MapTileGrid.tileIdFor(latitude, longitude),
            datasetVersion = "bundled-v1"
        )
    }
}
