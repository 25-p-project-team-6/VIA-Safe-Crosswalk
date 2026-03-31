package kr.co.gachon.pproject6.via.map

import android.content.Context
import android.util.Log
import kr.co.gachon.pproject6.via.BuildConfig
import kr.co.gachon.pproject6.via.context.SimpleJsonParser
import kr.co.gachon.pproject6.via.context.jsonDoubleOrNull
import kr.co.gachon.pproject6.via.context.jsonObjectOrNull
import kr.co.gachon.pproject6.via.context.jsonStringOrNull
import kr.co.gachon.pproject6.via.onboarding.AppPreferences
import java.io.BufferedWriter
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL
import java.nio.charset.StandardCharsets
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.TimeZone
import java.util.UUID
import kotlin.concurrent.thread

data class KineticGuestSession(
    val guestSessionId: String,
    val sessionExpiresAtIso: String?,
    val tokenType: String?,
    val tileUrlTemplate: String,
    val expiresAtIso: String?,
    val minZoom: Int,
    val maxZoom: Int,
    val tileSize: Int,
    val style: String?,
    val expiresAtEpochMs: Long?
) {
    fun isExpiredSoon(
        nowEpochMs: Long,
        refreshBufferMs: Long = 60_000L
    ): Boolean {
        val expiry = expiresAtEpochMs ?: return true
        return nowEpochMs + refreshBufferMs >= expiry
    }

    fun tileUrlFor(
        zoom: Int,
        x: Int,
        y: Int
    ): String {
        return tileUrlTemplate
            .replace("{z}", zoom.toString())
            .replace("{x}", x.toString())
            .replace("{y}", y.toString())
    }
}

class KineticGuestSessionManager private constructor(
    context: Context,
    private val preferences: AppPreferences = AppPreferences(context),
    private val timeProvider: () -> Long = System::currentTimeMillis
) {
    private val appContext = context.applicationContext

    @Volatile
    private var activeSession: KineticGuestSession? = null

    fun prefetchIfNeeded() {
        thread(start = true, isDaemon = true, name = "kinetic-map-prefetch") {
            runCatching { getValidSession(forceRefresh = false) }
                .onFailure { Log.w("KineticMap", "Guest session prefetch failed", it) }
        }
    }

    @Synchronized
    fun getValidSession(forceRefresh: Boolean): KineticGuestSession {
        val now = timeProvider()
        val current = activeSession
        if (!forceRefresh && current != null && !current.isExpiredSoon(now)) {
            return current
        }
        val refreshed = requestGuestSession()
        activeSession = refreshed
        Log.i(
            "KineticMap",
            "Guest session ready id=${refreshed.guestSessionId} exp=${refreshed.expiresAtIso ?: "unknown"}"
        )
        return refreshed
    }

    @Synchronized
    fun peekValidSession(): KineticGuestSession? {
        val current = activeSession ?: return null
        return if (current.isExpiredSoon(timeProvider())) {
            null
        } else {
            current
        }
    }

    @Synchronized
    fun invalidateSession() {
        activeSession = null
    }

    fun installationId(): String {
        val existing = preferences.mapInstallationId
        if (!existing.isNullOrBlank()) {
            return existing
        }
        val generated = UUID.randomUUID().toString()
        preferences.mapInstallationId = generated
        return generated
    }

    private fun requestGuestSession(): KineticGuestSession {
        val url = URL("${BuildConfig.KINETIC_MAP_API_BASE_URL}/v1/guest-sessions")
        val payload =
            """
            {
              "device_id": "${escapeJson(installationId())}",
              "platform": "android",
              "app_version": "${escapeJson(BuildConfig.VERSION_NAME)}",
              "style": "${escapeJson(BuildConfig.KINETIC_MAP_STYLE)}"
            }
            """.trimIndent()

        val connection = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = NETWORK_TIMEOUT_MS
            readTimeout = NETWORK_TIMEOUT_MS
            doOutput = true
            setRequestProperty("Content-Type", "application/json")
            setRequestProperty("Accept", "application/json")
        }

        try {
            BufferedWriter(OutputStreamWriter(connection.outputStream, StandardCharsets.UTF_8)).use {
                it.write(payload)
            }

            val statusCode = connection.responseCode
            val body =
                (if (statusCode in 200..299) connection.inputStream else connection.errorStream)
                    ?.bufferedReader(StandardCharsets.UTF_8)
                    ?.use { it.readText() }
                    .orEmpty()

            if (statusCode !in 200..299) {
                throw IllegalStateException("Guest session request failed with HTTP $statusCode: $body")
            }
            return parseGuestSessionResponse(body)
        } finally {
            connection.disconnect()
        }
    }

    companion object {
        private const val NETWORK_TIMEOUT_MS = 7_500

        @Volatile
        private var instance: KineticGuestSessionManager? = null

        fun from(context: Context): KineticGuestSessionManager {
            return instance ?: synchronized(this) {
                instance ?: KineticGuestSessionManager(context).also { instance = it }
            }
        }
    }
}

internal fun parseGuestSessionResponse(json: String): KineticGuestSession {
    val root = SimpleJsonParser.parseObject(json)
    val guestSessionId = root["guest_session_id"].jsonStringOrNull().orEmpty()
    val tileUrlTemplate = root["tile_url_template"].jsonStringOrNull().orEmpty()
    require(guestSessionId.isNotBlank()) { "guest_session_id missing" }
    require(tileUrlTemplate.isNotBlank()) { "tile_url_template missing" }

    val expiresAtIso = root["expires_at"].jsonStringOrNull()
    return KineticGuestSession(
        guestSessionId = guestSessionId,
        sessionExpiresAtIso = root["session_expires_at"].jsonStringOrNull(),
        tokenType = root["token_type"].jsonStringOrNull(),
        tileUrlTemplate = tileUrlTemplate,
        expiresAtIso = expiresAtIso,
        minZoom = (root["min_zoom"].jsonDoubleOrNull() ?: 0.0).toInt(),
        maxZoom = (root["max_zoom"].jsonDoubleOrNull() ?: 19.0).toInt(),
        tileSize = (root["tile_size"].jsonDoubleOrNull() ?: 256.0).toInt(),
        style = root["style"].jsonStringOrNull(),
        expiresAtEpochMs = expiresAtIso?.let(::parseIsoUtcOrNull)
    )
}

private fun parseIsoUtcOrNull(value: String): Long? {
    return runCatching {
        val formatter = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US)
        formatter.timeZone = TimeZone.getTimeZone("UTC")
        formatter.parse(value)?.time
    }.getOrNull()
}

private fun escapeJson(value: String): String {
    return value
        .replace("\\", "\\\\")
        .replace("\"", "\\\"")
}
