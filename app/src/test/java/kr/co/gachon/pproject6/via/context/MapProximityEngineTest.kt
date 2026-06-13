package kr.co.gachon.pproject6.via.context

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class MapProximityEngineTest {
    @Test
    fun requiresTwoConsecutiveAcceptedFixesBeforeEnteringNearState() {
        val engine = MapProximityEngine()
        val cluster = buildCrosswalkClusters(listOf(crosswalkFeature(id = "crosswalk-a", latitude = 37.4500, longitude = 127.1280))).single()
        val point = GeoPoint(latitude = 37.45005, longitude = 127.1280)

        val first = engine.update(point, 5f, null, listOf(cluster), "bundled-v1", false)
        val second = engine.update(point, 5f, null, listOf(cluster), "bundled-v1", false)

        assertFalse(first.isNearKnownFeature)
        assertTrue(second.isNearKnownFeature)
        assertEquals("crosswalk-a", second.matchedClusterId)
        assertEquals("crosswalk-a", second.matchedAnchorId)
        assertEquals(1, second.matchedMemberCount)
    }

    @Test
    fun overlappingBundledEndpointsAndOsmCenterBecomeOneCluster() {
        val firstEndpoint = crosswalkFeature(id = "bundled-left", latitude = 37.45000, longitude = 127.12800)
        val secondEndpoint = crosswalkFeature(id = "bundled-right", latitude = 37.45008, longitude = 127.12800)
        val osmCenter =
            crosswalkFeature(
                id = "osm-center",
                latitude = 37.45004,
                longitude = 127.12800,
                source = MapFeatureSource.OSM
            )

        val clusters = buildCrosswalkClusters(listOf(firstEndpoint, secondEndpoint, osmCenter))

        assertEquals(1, clusters.size)
        assertEquals(3, clusters.single().memberCount)
        assertEquals(MapFeatureSource.HYBRID, clusters.single().source)
    }

    @Test
    fun preservesClusterMatchWhenAnchorPointChangesInsideSameCluster() {
        val engine = MapProximityEngine()
        val features =
            listOf(
                crosswalkFeature(id = "bundled-left", latitude = 37.45000, longitude = 127.12800),
                crosswalkFeature(id = "bundled-right", latitude = 37.45008, longitude = 127.12800),
                crosswalkFeature(id = "osm-center", latitude = 37.45004, longitude = 127.12800, source = MapFeatureSource.OSM)
            )
        val cluster = buildCrosswalkClusters(features).single()
        val firstPoint = GeoPoint(latitude = 37.45001, longitude = 127.12800)
        val secondPoint = GeoPoint(latitude = 37.45007, longitude = 127.12800)

        engine.update(firstPoint, 4f, null, listOf(cluster), "bundled-v1+osm", false)
        val matched = engine.update(firstPoint, 4f, null, listOf(cluster), "bundled-v1+osm", false)
        val retained = engine.update(secondPoint, 4f, null, listOf(cluster), "bundled-v1+osm", false)

        assertEquals(matched.matchedClusterId, retained.matchedClusterId)
        assertEquals(MapClusterTransitionKind.NONE, retained.clusterTransitionKind)
    }

    @Test
    fun switchesToCloserClusterAfterTwoConsistentFixes() {
        val engine = MapProximityEngine()
        val first = buildCrosswalkClusters(listOf(crosswalkFeature(id = "cluster-a", latitude = 37.45000, longitude = 127.12800))).single()
        val second = buildCrosswalkClusters(listOf(crosswalkFeature(id = "cluster-b", latitude = 37.45018, longitude = 127.12800))).single()
        val initialPoint = GeoPoint(latitude = 37.45003, longitude = 127.12800)
        val movedPoint = GeoPoint(latitude = 37.45015, longitude = 127.12800)

        engine.update(initialPoint, 4f, null, listOf(first, second), "bundled-v1", false)
        engine.update(initialPoint, 4f, null, listOf(first, second), "bundled-v1", false)
        val firstSwitchAttempt = engine.update(movedPoint, 4f, null, listOf(first, second), "bundled-v1", false)
        val confirmedSwitch = engine.update(movedPoint, 4f, null, listOf(first, second), "bundled-v1", false)

        assertEquals("cluster-a", firstSwitchAttempt.matchedClusterId)
        assertEquals("cluster-b", confirmedSwitch.matchedClusterId)
    }

    @Test
    fun poorAccuracyDoesNotCreateNewMatch() {
        val engine = MapProximityEngine()
        val cluster = buildCrosswalkClusters(listOf(crosswalkFeature(id = "crosswalk-a", latitude = 37.4500, longitude = 127.1280))).single()
        val point = GeoPoint(latitude = 37.45005, longitude = 127.1280)

        val snapshot = engine.update(point, 40f, null, listOf(cluster), "bundled-v1", false)

        assertFalse(snapshot.isNearKnownFeature)
        assertEquals("bundled-v1", snapshot.datasetVersion)
    }

    @Test
    fun headingBreaksNearDistanceTiesWhenApproachBearingExists() {
        val engine = MapProximityEngine()
        val northSouth =
            buildCrosswalkClusters(
                listOf(
                    crosswalkFeature(
                        id = "crosswalk-ns",
                        latitude = 37.4500,
                        longitude = 127.1280,
                        approachBearings = listOf(0f, 180f)
                    )
                )
            ).single()
        val eastWest =
            buildCrosswalkClusters(
                listOf(
                    crosswalkFeature(
                        id = "crosswalk-ew",
                        latitude = 37.45002,
                        longitude = 127.1280,
                        approachBearings = listOf(90f, 270f)
                    )
                )
            ).single()
        val point = GeoPoint(latitude = 37.45008, longitude = 127.1280)

        engine.update(point, 5f, 92f, listOf(northSouth, eastWest), "bundled-v1", false)
        val snapshot = engine.update(point, 5f, 92f, listOf(northSouth, eastWest), "bundled-v1", false)

        assertTrue(snapshot.isNearKnownFeature)
        assertEquals("crosswalk-ew", snapshot.matchedClusterId)
    }

    private fun crosswalkFeature(
        id: String,
        latitude: Double,
        longitude: Double,
        approachBearings: List<Float> = emptyList(),
        kind: MapFeatureKind = MapFeatureKind.CROSSWALK,
        source: MapFeatureSource = MapFeatureSource.BUNDLED
    ): MapFeatureRecord {
        return MapFeatureRecord(
            id = id,
            kind = kind,
            point = GeoPoint(latitude = latitude, longitude = longitude),
            triggerRadiusMeters = 35f,
            exitRadiusMeters = 55f,
            approachBearings = approachBearings,
            regionTileId = MapTileGrid.tileIdFor(latitude, longitude),
            datasetVersion = "bundled-v1",
            source = source
        )
    }
}
