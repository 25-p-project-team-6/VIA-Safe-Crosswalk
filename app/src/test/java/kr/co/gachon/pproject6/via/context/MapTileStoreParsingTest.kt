package kr.co.gachon.pproject6.via.context

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MapTileStoreParsingTest {
    @Test
    fun neighboringTileIdsReturnsCenterAndEightNeighbors() {
        val tileIds = MapTileGrid.neighboringTileIds(GeoPoint(37.4501, 127.1287))

        assertEquals(9, tileIds.size)
        assertTrue(tileIds.contains("3745_12712"))
        assertTrue(tileIds.contains("3744_12711"))
        assertTrue(tileIds.contains("3746_12713"))
    }

    @Test
    fun parsesManifestAndTileJsonIntoFeatureRecords() {
        val manifest =
            parseMapDatasetManifest(
                """
                {
                  "version": "remote-v2",
                  "tiles": [
                    {
                      "tileId": "3745_12712",
                      "file": "tiles/3745_12712.json",
                      "checksum": "abc",
                      "downloadUrl": "https://example.com/3745_12712.json"
                    }
                  ]
                }
                """.trimIndent()
            )
        val features =
            parseMapTileFeatures(
                tileId = "3745_12712",
                datasetVersion = manifest.version,
                json =
                    """
                    {
                      "features": [
                        {
                          "id": "crosswalk-a",
                          "kind": "crosswalk",
                          "lat": 37.4501,
                          "lon": 127.1287,
                          "triggerRadiusMeters": 30,
                          "approachBearings": [90, 270]
                        }
                      ]
                    }
                    """.trimIndent()
            )

        assertEquals("remote-v2", manifest.version)
        assertEquals("tiles/3745_12712.json", manifest.tiles.getValue("3745_12712").filePath)
        assertEquals(1, features.size)
        assertEquals("crosswalk-a", features.single().id)
        assertEquals(30f, features.single().triggerRadiusMeters)
        assertEquals(55f, features.single().exitRadiusMeters)
        assertEquals(listOf(90f, 270f), features.single().approachBearings)
    }
}
