package kr.co.gachon.pproject6.via.map

import kr.co.gachon.pproject6.via.context.GeoPoint
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class OsmNearbyCrossingFetcherTest {
    @Test
    fun parseOverpassCrossingsKeepsSignalControlledFeaturePerLocation() {
        val currentPoint = GeoPoint(37.455279, 127.133574)
        val crossings =
            parseOverpassCrossings(
                """
                {
                  "elements": [
                    {
                      "type": "node",
                      "id": 1,
                      "lat": 37.455300,
                      "lon": 127.133500,
                      "tags": {
                        "highway": "crossing"
                      }
                    },
                    {
                      "type": "node",
                      "id": 2,
                      "lat": 37.4553001,
                      "lon": 127.1335001,
                      "tags": {
                        "highway": "crossing",
                        "crossing": "traffic_signals"
                      }
                    },
                    {
                      "type": "way",
                      "id": 3,
                      "center": {
                        "lat": 37.455800,
                        "lon": 127.133900
                      },
                      "tags": {
                        "highway": "footway",
                        "footway": "crossing"
                      }
                    }
                  ]
                }
                """.trimIndent(),
                currentPoint
            )

        assertEquals(2, crossings.size)
        assertTrue(crossings.any { it.kind == "osm_signal_crossing" })
        assertTrue(crossings.any { it.kind == "osm_crossing_way" })
    }
}
