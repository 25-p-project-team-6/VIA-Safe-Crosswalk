package kr.co.gachon.pproject6.via.map

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class KineticGuestSessionManagerTest {
    @Test
    fun parseGuestSessionResponseExtractsTileTemplateAndExpiry() {
        val session =
            parseGuestSessionResponse(
                """
                {
                  "guest_session_id": "gst_test",
                  "session_expires_at": "2026-03-29T08:00:00Z",
                  "token_type": "Guest",
                  "tile_url_template": "https://tile.map.kinetic.moe/v1/tiles/{z}/{x}/{y}.png?tile_token=abc",
                  "expires_at": "2026-03-29T07:30:00Z",
                  "min_zoom": 0,
                  "max_zoom": 19,
                  "tile_size": 256,
                  "style": "default"
                }
                """.trimIndent()
            )

        assertEquals("gst_test", session.guestSessionId)
        assertEquals(
            "https://tile.map.kinetic.moe/v1/tiles/5/10/12.png?tile_token=abc",
            session.tileUrlFor(5, 10, 12)
        )
        assertEquals(0, session.minZoom)
        assertEquals(19, session.maxZoom)
        assertEquals(256, session.tileSize)
        assertEquals("default", session.style)
        assertFalse(session.isExpiredSoon(0L))
    }

    @Test
    fun isExpiredSoonUsesRefreshBuffer() {
        val session =
            parseGuestSessionResponse(
                """
                {
                  "guest_session_id": "gst_test",
                  "tile_url_template": "https://tile.map.kinetic.moe/v1/tiles/{z}/{x}/{y}.png?tile_token=abc",
                  "expires_at": "2026-03-29T07:30:00Z",
                  "min_zoom": 0,
                  "max_zoom": 19,
                  "tile_size": 256
                }
                """.trimIndent()
            )

        val justBeforeExpiry = session.expiresAtEpochMs!! - 30_000L
        val wellBeforeExpiry = session.expiresAtEpochMs!! - 120_000L

        assertTrue(session.isExpiredSoon(justBeforeExpiry))
        assertFalse(session.isExpiredSoon(wellBeforeExpiry))
    }
}
