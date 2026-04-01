package kr.co.gachon.pproject6.via

import android.Manifest
import android.annotation.SuppressLint
import android.content.ActivityNotFoundException
import android.content.Intent
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.net.Uri
import android.os.Bundle
import android.os.Looper
import android.view.View
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.google.android.material.button.MaterialButton
import com.google.android.material.floatingactionbutton.FloatingActionButton
import kr.co.gachon.pproject6.via.context.GeoPoint
import kr.co.gachon.pproject6.via.context.MapProximityManager
import kr.co.gachon.pproject6.via.context.MapTileStore
import kr.co.gachon.pproject6.via.context.haversineDistanceMeters
import kr.co.gachon.pproject6.via.map.KineticGuestSession
import kr.co.gachon.pproject6.via.map.KineticGuestSessionManager
import kr.co.gachon.pproject6.via.map.MapDebugCacheManager
import kr.co.gachon.pproject6.via.map.OsmNearbyCrossing
import kr.co.gachon.pproject6.via.map.OsmNearbyCrossingFetcher
import java.io.ByteArrayInputStream
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.nio.charset.StandardCharsets
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

private const val OSM_RADIUS_METERS = 400
private const val OSM_LIMIT = 100
private const val BUNDLED_LOAD_RADIUS_METERS = 700

class DebugMapActivity : AppCompatActivity() {
    private lateinit var rootView: View
    private lateinit var summaryTextView: TextView
    private lateinit var nearbyTextView: TextView
    private lateinit var webView: WebView
    private lateinit var externalMapButton: MaterialButton
    private lateinit var recenterMapButton: FloatingActionButton
    private lateinit var mapState: DebugMapState
    private lateinit var sessionManager: KineticGuestSessionManager
    private lateinit var locationManager: LocationManager
    private lateinit var osmFetcher: OsmNearbyCrossingFetcher
    private lateinit var mapProximityManager: MapProximityManager
    private val tileRefreshInFlight = AtomicBoolean(false)
    private var activeSession: KineticGuestSession? = null
    private var currentMarkerInjected = false
    private var lastDataRefreshAtMs: Long = 0L
    private var lastDataRefreshPoint: GeoPoint? = null
    private val locationListener = LocationListener { location ->
        handleLiveLocation(location)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_debug_map)

        rootView = findViewById(R.id.debugMapRoot)
        summaryTextView = findViewById(R.id.debugMapSummaryText)
        nearbyTextView = findViewById(R.id.debugMapNearbyText)
        webView = findViewById(R.id.debugMapWebView)
        externalMapButton = findViewById(R.id.openExternalMapButton)
        recenterMapButton = findViewById(R.id.recenterMapButton)

        sessionManager = KineticGuestSessionManager.from(this)
        locationManager = getSystemService(LocationManager::class.java)
        osmFetcher = OsmNearbyCrossingFetcher(this)
        mapProximityManager = MapProximityManager(this)
        mapState = DebugMapState.fromIntent(intent, this)

        title = "지도 디버그"
        summaryTextView.text = mapState.toHeaderSummaryText()
        nearbyTextView.text = mapState.toNearbySummaryText()
        externalMapButton.text = "구글 지도에서 현재 위치 보기"

        applyInsets()
        configureWebView()

        externalMapButton.setOnClickListener {
            openExternalMap(mapState)
        }
        recenterMapButton.setOnClickListener {
            recenterMapOnCurrentLocation()
        }

        val cachedSession = sessionManager.peekValidSession()
        if (cachedSession != null) {
            activeSession = cachedSession
            summaryTextView.text = mapState.toHeaderSummaryText()
            nearbyTextView.text = mapState.toNearbySummaryText()
            webView.loadDataWithBaseURL(
                "file:///android_asset/",
                mapState.toLeafletHtml(cachedSession),
                "text/html",
                "utf-8",
                null
            )
            loadDebugDataForCurrentLocation(mapState.currentPoint())
        } else {
            loadMap(forceRefresh = false)
        }
    }

    override fun onResume() {
        super.onResume()
        startLocationUpdates()
    }

    override fun onPause() {
        stopLocationUpdates()
        super.onPause()
    }

    override fun onDestroy() {
        webView.destroy()
        super.onDestroy()
    }

    private fun applyInsets() {
        val basePaddingLeft = rootView.paddingLeft
        val basePaddingTop = rootView.paddingTop
        val basePaddingRight = rootView.paddingRight
        val basePaddingBottom = rootView.paddingBottom
        ViewCompat.setOnApplyWindowInsetsListener(rootView) { view, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            view.setPadding(
                basePaddingLeft,
                basePaddingTop + bars.top,
                basePaddingRight,
                basePaddingBottom + bars.bottom
            )
            insets
        }
    }

    @SuppressLint("SetJavaScriptEnabled")
    private fun configureWebView() {
        webView.settings.javaScriptEnabled = true
        webView.settings.domStorageEnabled = true
        webView.webViewClient = KineticTileWebViewClient()
    }

    @SuppressLint("MissingPermission")
    private fun startLocationUpdates() {
        if (!hasLocationPermission()) {
            return
        }
        seedLastKnownLocation()
        runCatching {
            if (locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
                locationManager.requestLocationUpdates(
                    LocationManager.GPS_PROVIDER,
                    1_000L,
                    1f,
                    locationListener,
                    Looper.getMainLooper()
                )
            }
        }
        runCatching {
            if (locationManager.isProviderEnabled(LocationManager.NETWORK_PROVIDER)) {
                locationManager.requestLocationUpdates(
                    LocationManager.NETWORK_PROVIDER,
                    1_500L,
                    2f,
                    locationListener,
                    Looper.getMainLooper()
                )
            }
        }
    }

    @SuppressLint("MissingPermission")
    private fun seedLastKnownLocation() {
        val candidates =
            buildList {
                runCatching {
                    if (locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
                        locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER)?.let { add(it) }
                    }
                }
                runCatching {
                    if (locationManager.isProviderEnabled(LocationManager.NETWORK_PROVIDER)) {
                        locationManager.getLastKnownLocation(LocationManager.NETWORK_PROVIDER)?.let { add(it) }
                    }
                }
            }
        val best =
            candidates.minWithOrNull(
                compareBy<Location> {
                    if (it.hasAccuracy()) it.accuracy else Float.MAX_VALUE
                }.thenByDescending { it.time }
            )
        if (best != null) {
            handleLiveLocation(best)
        }
    }

    private fun stopLocationUpdates() {
        runCatching { locationManager.removeUpdates(locationListener) }
    }

    private fun hasLocationPermission(): Boolean {
        val fine =
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED
        val coarse =
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_COARSE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED
        return fine || coarse
    }

    private fun loadMap(
        forceRefresh: Boolean,
        silent: Boolean = false
    ) {
        if (!silent) {
            summaryTextView.text = mapState.toHeaderSummaryText() + "\n\n지도 세션 요청 중..."
            nearbyTextView.text = mapState.toNearbySummaryText()
        }
        thread(start = true, isDaemon = true, name = "kinetic-map-load") {
            runCatching {
                sessionManager.getValidSession(forceRefresh = forceRefresh)
            }
                .onSuccess { session ->
                    runOnUiThread {
                        activeSession = session
                        tileRefreshInFlight.set(false)
                        currentMarkerInjected = false
                        summaryTextView.text = mapState.toHeaderSummaryText()
                        nearbyTextView.text = mapState.toNearbySummaryText()
                        webView.loadDataWithBaseURL(
                            "file:///android_asset/",
                            mapState.toLeafletHtml(session),
                            "text/html",
                            "utf-8",
                            null
                        )
                        loadDebugDataForCurrentLocation(mapState.currentPoint())
                    }
                }
                .onFailure { error ->
                    runOnUiThread {
                        activeSession = null
                        tileRefreshInFlight.set(false)
                        summaryTextView.text =
                            mapState.toHeaderSummaryText() + "\n\n지도 세션 요청 실패: ${error.message}"
                        nearbyTextView.text = mapState.toNearbySummaryText()
                        webView.loadDataWithBaseURL(
                            null,
                            mapState.toErrorHtml(error.message ?: "unknown error"),
                            "text/html",
                            "utf-8",
                            null
                        )
                    }
                }
        }
    }

    private fun handleLiveLocation(
        location: Location
    ) {
        mapProximityManager.onLocation(location, headingDegrees = null)
        val matchedSnapshot = mapProximityManager.snapshot()
        val updatedBase =
            mapState.base.copy(
                currentLat = location.latitude,
                currentLon = location.longitude,
                currentAccMeters = if (location.hasAccuracy()) location.accuracy else mapState.base.currentAccMeters,
                matchedKind = matchedSnapshot.matchedKind?.wireName,
                matchedSource = matchedSnapshot.matchedSource?.wireName,
                matchedId = matchedSnapshot.matchedFeatureId,
                matchedLat = matchedSnapshot.matchedLatitude,
                matchedLon = matchedSnapshot.matchedLongitude,
                matchedDistMeters = matchedSnapshot.distanceMeters,
                isNearKnownFeature = matchedSnapshot.isNearKnownFeature
            )
        mapState = mapState.copy(base = updatedBase)
        summaryTextView.text = mapState.toHeaderSummaryText()
        updateCurrentLocationOnMap(location.latitude, location.longitude, mapState.currentBundleRadiusMeters())

        val currentPoint = GeoPoint(location.latitude, location.longitude)
        val shouldRefreshData =
            lastDataRefreshPoint == null ||
                haversineDistanceMeters(lastDataRefreshPoint!!, currentPoint) >= 20f ||
                System.currentTimeMillis() - lastDataRefreshAtMs >= 5_000L
        if (shouldRefreshData) {
            lastDataRefreshPoint = currentPoint
            lastDataRefreshAtMs = System.currentTimeMillis()
            loadDebugDataForCurrentLocation(currentPoint)
        }
    }

    private fun loadDebugDataForCurrentLocation(
        currentPoint: GeoPoint
    ) {
        val previousState = mapState
        thread(start = true, isDaemon = true, name = "debug-map-data-refresh") {
            val bundledNearby = loadBundledNearby(currentPoint)
            val fetchedOsmNearby =
                osmFetcher.fetchNearby(
                    point = currentPoint,
                    radiusMeters = OSM_RADIUS_METERS,
                    limit = OSM_LIMIT
                )
            val osmNearby =
                if (
                    fetchedOsmNearby.isEmpty() &&
                        previousState.osmCrossings.isNotEmpty() &&
                        haversineDistanceMeters(previousState.currentPoint(), currentPoint) <= OSM_RADIUS_METERS
                ) {
                    previousState.osmCrossings
                } else {
                    fetchedOsmNearby
                }
            val refreshedState =
                mapState.withCurrentLocation(currentPoint).withNearbyData(
                    bundledNearby = bundledNearby,
                    osmNearby = osmNearby
                )
            runOnUiThread {
                mapState = refreshedState
                summaryTextView.text = mapState.toHeaderSummaryText()
                nearbyTextView.text = mapState.toNearbySummaryText()
                if (activeSession != null && currentMarkerInjected) {
                    pushMapDataUpdate()
                }
            }
        }
    }

    private fun loadBundledNearby(
        currentPoint: GeoPoint
    ): List<NearbyFeature> {
        return runCatching {
            val store = MapTileStore(this)
            val loadResult = store.loadWithinRadius(currentPoint, BUNDLED_LOAD_RADIUS_METERS)
            loadResult.features
                .distinctBy { it.id }
                .map {
                    NearbyFeature(
                        id = it.id,
                        kind = it.kind.wireName,
                        lat = it.point.latitude,
                        lon = it.point.longitude,
                        distanceMeters = haversineDistanceMeters(currentPoint, it.point)
                    )
                }
                .sortedBy { it.distanceMeters }
        }.getOrDefault(emptyList())
    }

    private fun updateCurrentLocationOnMap(
        latitude: Double,
        longitude: Double,
        bundledRadiusMeters: Int
    ) {
        val script =
            "window.updateCurrentLocation && window.updateCurrentLocation($latitude,$longitude,250,$bundledRadiusMeters);"
        if (currentMarkerInjected) {
            webView.evaluateJavascript(script, null)
        }
    }

    private fun pushMapDataUpdate() {
        if (!currentMarkerInjected) {
            return
        }
        webView.evaluateJavascript(mapState.toLeafletUpdateScript(), null)
    }

    private fun recenterMapOnCurrentLocation() {
        if (!currentMarkerInjected) {
            Toast.makeText(this, "지도를 아직 불러오는 중입니다", Toast.LENGTH_SHORT).show()
            return
        }
        webView.evaluateJavascript(
            "window.recenterToCurrentLocation && window.recenterToCurrentLocation();",
            null
        )
    }

    private fun openExternalMap(state: DebugMapState) {
        val candidates = state.toExternalMapIntents()
        for (intent in candidates) {
            try {
                startActivity(intent)
                return
            } catch (_: ActivityNotFoundException) {
                // try next candidate
            }
        }
        Toast.makeText(this, "지도를 열 수 있는 앱이 없습니다", Toast.LENGTH_SHORT).show()
    }

    private fun triggerTileSessionRefresh() {
        if (!tileRefreshInFlight.compareAndSet(false, true)) {
            return
        }
        sessionManager.invalidateSession()
        runOnUiThread {
            loadMap(forceRefresh = true, silent = true)
        }
    }

    private inner class KineticTileWebViewClient : WebViewClient() {
        override fun onPageFinished(view: WebView?, url: String?) {
            super.onPageFinished(view, url)
            currentMarkerInjected = true
            updateCurrentLocationOnMap(
                mapState.base.currentLat,
                mapState.base.currentLon,
                mapState.currentBundleRadiusMeters()
            )
            pushMapDataUpdate()
        }

        override fun shouldInterceptRequest(
            view: WebView?,
            request: WebResourceRequest
        ): WebResourceResponse? {
            val url = request.url.toString()
            if (!url.startsWith("https://tile.map.kinetic.moe/")) {
                return super.shouldInterceptRequest(view, request)
            }

            return fetchTileResponse(url)
        }

        private fun fetchTileResponse(url: String): WebResourceResponse {
            val cached = this@DebugMapActivity.readCachedTile(url)
            if (cached != null) {
                return cached
            }

            val connection = (URL(url).openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
                connectTimeout = 7_500
                readTimeout = 7_500
                instanceFollowRedirects = true
                setRequestProperty(
                    "User-Agent",
                    "VIA-DebugMap/${BuildConfig.VERSION_NAME} (Android)"
                )
                setRequestProperty("Accept", "image/png,image/*;q=0.8,*/*;q=0.5")
            }
            var statusCode: Int? = null

            return try {
                statusCode = connection.responseCode
                when (statusCode) {
                    HttpURLConnection.HTTP_OK -> {
                        val mimeType = connection.contentType?.substringBefore(";") ?: "image/png"
                        val bytes = connection.inputStream.use { it.readBytes() }
                        this@DebugMapActivity.writeCachedTile(
                            url = url,
                            payload = bytes,
                            mimeType = mimeType,
                            encoding = connection.contentEncoding ?: "binary",
                            expiryEpochMs = parseExpiryEpochMs(connection)
                        )
                        WebResourceResponse(
                            mimeType,
                            connection.contentEncoding ?: "binary",
                            statusCode ?: HttpURLConnection.HTTP_OK,
                            connection.responseMessage ?: "OK",
                            connection.headerFields.filterKeys { it != null }.mapValues { it.value.joinToString(",") },
                            ByteArrayInputStream(bytes)
                        )
                    }

                    HttpURLConnection.HTTP_UNAUTHORIZED -> {
                        triggerTileSessionRefresh()
                        blankTileResponse()
                    }

                    HttpURLConnection.HTTP_NOT_FOUND -> blankTileResponse()
                    else -> blankTileResponse()
                }
            } catch (_: Exception) {
                blankTileResponse()
            } finally {
                if (statusCode != HttpURLConnection.HTTP_OK) {
                    connection.disconnect()
                }
            }
        }
    }

    companion object {
        private const val EXTRA_CURRENT_LAT = "current_lat"
        private const val EXTRA_CURRENT_LON = "current_lon"
        private const val EXTRA_CURRENT_ACC = "current_acc"
        private const val EXTRA_MATCHED_LAT = "matched_lat"
        private const val EXTRA_MATCHED_LON = "matched_lon"
        private const val EXTRA_MATCHED_KIND = "matched_kind"
        private const val EXTRA_MATCHED_SOURCE = "matched_source"
        private const val EXTRA_MATCHED_ID = "matched_id"
        private const val EXTRA_MATCHED_DIST = "matched_dist"
        private const val EXTRA_MAP_VERSION = "map_version"
        private const val EXTRA_MAP_NEAR = "map_near"

        fun newIntent(
            activity: AppCompatActivity,
            currentLat: Double,
            currentLon: Double,
            currentAccMeters: Float?,
            matchedLat: Double?,
            matchedLon: Double?,
            matchedKind: String?,
            matchedSource: String?,
            matchedId: String?,
            matchedDistMeters: Float?,
            mapVersion: String?,
            isNearKnownFeature: Boolean
        ): Intent {
            return Intent(activity, DebugMapActivity::class.java).apply {
                putExtra(EXTRA_CURRENT_LAT, currentLat)
                putExtra(EXTRA_CURRENT_LON, currentLon)
                currentAccMeters?.let { putExtra(EXTRA_CURRENT_ACC, it) }
                matchedLat?.let { putExtra(EXTRA_MATCHED_LAT, it) }
                matchedLon?.let { putExtra(EXTRA_MATCHED_LON, it) }
                putExtra(EXTRA_MATCHED_KIND, matchedKind)
                putExtra(EXTRA_MATCHED_SOURCE, matchedSource)
                putExtra(EXTRA_MATCHED_ID, matchedId)
                matchedDistMeters?.let { putExtra(EXTRA_MATCHED_DIST, it) }
                putExtra(EXTRA_MAP_VERSION, mapVersion)
                putExtra(EXTRA_MAP_NEAR, isNearKnownFeature)
            }
        }

        internal fun readIntent(
            intent: Intent
        ): BaseMapDebugState {
            return BaseMapDebugState(
                currentLat = intent.getDoubleExtra(EXTRA_CURRENT_LAT, 0.0),
                currentLon = intent.getDoubleExtra(EXTRA_CURRENT_LON, 0.0),
                currentAccMeters =
                    intent.extras?.takeIf { it.containsKey(EXTRA_CURRENT_ACC) }?.getFloat(EXTRA_CURRENT_ACC),
                matchedLat =
                    intent.extras?.takeIf { it.containsKey(EXTRA_MATCHED_LAT) }?.getDouble(EXTRA_MATCHED_LAT),
                matchedLon =
                    intent.extras?.takeIf { it.containsKey(EXTRA_MATCHED_LON) }?.getDouble(EXTRA_MATCHED_LON),
                matchedKind = intent.getStringExtra(EXTRA_MATCHED_KIND),
                matchedSource = intent.getStringExtra(EXTRA_MATCHED_SOURCE),
                matchedId = intent.getStringExtra(EXTRA_MATCHED_ID),
                matchedDistMeters =
                    intent.extras?.takeIf { it.containsKey(EXTRA_MATCHED_DIST) }?.getFloat(EXTRA_MATCHED_DIST),
                mapVersion = intent.getStringExtra(EXTRA_MAP_VERSION),
                isNearKnownFeature = intent.getBooleanExtra(EXTRA_MAP_NEAR, false)
            )
        }
    }
}

internal data class BaseMapDebugState(
    val currentLat: Double,
    val currentLon: Double,
    val currentAccMeters: Float?,
    val matchedLat: Double?,
    val matchedLon: Double?,
    val matchedKind: String?,
    val matchedSource: String?,
    val matchedId: String?,
    val matchedDistMeters: Float?,
    val mapVersion: String?,
    val isNearKnownFeature: Boolean
)

private data class DebugMapState(
    val base: BaseMapDebugState,
    val nearbyFeatures: List<NearbyFeature>,
    val osmCrossings: List<OsmNearbyCrossing> = emptyList()
) {
    fun currentBundleRadiusMeters(): Int = BUNDLED_LOAD_RADIUS_METERS
    fun currentOsmRadiusMeters(): Int = OSM_RADIUS_METERS
    fun currentPoint(): GeoPoint = GeoPoint(base.currentLat, base.currentLon)

    fun toHeaderSummaryText(): String {
        val bundledNearest =
            nearbyFeatures.minByOrNull { it.distanceMeters }?.distanceMeters
        val osmNearest =
            osmCrossings.minByOrNull { it.distanceMeters }?.distanceMeters
        val matchedSummary =
            if (base.matchedKind != null && base.matchedDistMeters != null) {
                "${base.matchedKind}/${base.matchedSource ?: "unknown"} · ${String.format(Locale.US, "%.1f m", base.matchedDistMeters)}"
            } else {
                "none"
            }
        return buildString {
            appendLine("Current  : ${formatCoord(base.currentLat)}, ${formatCoord(base.currentLon)}")
            appendLine("Accuracy : ${base.currentAccMeters?.let { String.format(Locale.US, "%.1f m", it) } ?: "n/a"}")
            appendLine("Matched  : $matchedSummary")
            appendLine(
                "Nearest B: ${
                    bundledNearest?.let { String.format(Locale.US, "%.1f m", it) } ?: "n/a"
                }"
            )
            appendLine(
                "Bundled  : ${nearbyFeatures.size}개"
            )
            appendLine(
                "Nearest O: ${
                    osmNearest?.let { String.format(Locale.US, "%.1f m", it) } ?: "n/a"
                }"
            )
            appendLine(
                "OSM      : ${osmCrossings.size}개"
            )
            append("Radius   : OSM ${currentOsmRadiusMeters()} m / Bundled tile load")
        }
    }

    fun toNearbySummaryText(): String {
        return buildString {
            appendLine("Bundled Nearby (${nearbyFeatures.size})")
            if (nearbyFeatures.isEmpty()) {
                appendLine("none")
            } else {
                nearbyFeatures.forEachIndexed { index, feature ->
                    appendLine(
                        "${index + 1}. ${feature.kind} ${String.format(Locale.US, "%.1f", feature.distanceMeters)}m ${shortId(feature.id)}"
                    )
                }
            }
            appendLine()
            appendLine("OSM Nearby (${osmCrossings.size})")
            if (osmCrossings.isEmpty()) {
                append("none")
            } else {
                osmCrossings.forEachIndexed { index, feature ->
                    appendLine(
                        "${index + 1}. ${feature.kind} ${String.format(Locale.US, "%.1f", feature.distanceMeters)}m ${shortId(feature.id)}"
                    )
                }
            }
        }
    }

    fun toLeafletUpdateScript(): String {
        return """
            window.updateDebugMapData && window.updateDebugMapData({
              current: { lat: ${base.currentLat}, lon: ${base.currentLon} },
              bundled: ${bundledFeaturesJs()},
              osm: ${osmFeaturesJs()},
              matched: ${matchedFeatureJs()},
              osmRadius: ${currentOsmRadiusMeters()},
              bundledRadius: ${currentBundleRadiusMeters()}
            });
        """.trimIndent()
    }

    fun toLeafletHtml(session: KineticGuestSession): String {
        return """
            <!doctype html>
            <html lang="ko">
            <head>
              <meta charset="utf-8" />
              <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes" />
              <link rel="stylesheet" href="leaflet/leaflet.css" />
              <style>
                html, body, #map { height: 100%; margin: 0; padding: 0; background: #101418; }
                .leaflet-control-attribution { font-size: 10px; }
              </style>
            </head>
            <body>
              <div id="map"></div>
              <script src="leaflet/leaflet.js"></script>
              <script>
                const map = L.map('map', { zoomControl: true, preferCanvas: true });
                const tileLayer = L.tileLayer(${jsString(session.tileUrlTemplate)}, {
                  minZoom: ${session.minZoom},
                  maxZoom: ${session.maxZoom},
                  tileSize: ${session.tileSize},
                  attribution: '© OpenStreetMap contributors / kinetic.moe'
                }).addTo(map);
                let currentLocation = { lat: ${base.currentLat}, lon: ${base.currentLon} };
                let currentOsmRadius = ${currentOsmRadiusMeters()};
                let currentBundledRadius = ${currentBundleRadiusMeters()};
                let currentMarker = L.circleMarker([${base.currentLat}, ${base.currentLon}], {
                  radius: 9,
                  color: '#2196f3',
                  fillColor: '#42a5f5',
                  fillOpacity: 0.95,
                  weight: 3
                }).addTo(map).bindPopup('현재 GPS').openPopup();
                let osmRadiusCircle = L.circle([${base.currentLat}, ${base.currentLon}], {
                  radius: ${currentOsmRadiusMeters()},
                  color: '#64b5f6',
                  weight: 2,
                  fillColor: '#64b5f6',
                  fillOpacity: 0.08
                }).addTo(map);
                let bundledFeatures = ${bundledFeaturesJs()};
                let osmFeatures = ${osmFeaturesJs()};
                let matchedFeature = ${matchedFeatureJs()};
                const bundledLayers = [];
                const osmLayers = [];
                let matchedMarker = null;
                let matchedLine = null;
                function distanceMeters(lat1, lon1, lat2, lon2) {
                  const R = 6371000;
                  const toRad = (v) => v * Math.PI / 180;
                  const dLat = toRad(lat2 - lat1);
                  const dLon = toRad(lon2 - lon1);
                  const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) *
                    Math.sin(dLon/2) * Math.sin(dLon/2);
                  return 2 * R * Math.asin(Math.min(1, Math.sqrt(a)));
                }
                function ensureBundledLayers() {
                  if (bundledLayers.length > 0) return;
                  bundledFeatures.forEach((feature) => {
                    const marker = L.circleMarker([feature.lat, feature.lon], {radius: 7, color: feature.stroke, fillColor: feature.fill, fillOpacity: 0.9, weight: 2})
                      .bindPopup(feature.popup);
                    bundledLayers.push({ feature, layer: marker, visible: false });
                  });
                }
                function clearBundledLayers() {
                  while (bundledLayers.length > 0) {
                    const entry = bundledLayers.pop();
                    if (entry.visible) {
                      map.removeLayer(entry.layer);
                    }
                  }
                }
                function ensureOsmLayers() {
                  if (osmLayers.length > 0) return;
                  osmFeatures.forEach((feature) => {
                    const group = L.layerGroup();
                    if (feature.geometry && feature.geometry.length >= 2) {
                      L.polyline(feature.geometry, {color: feature.stroke, weight: 4, opacity: 0.9}).addTo(group);
                    }
                    L.circleMarker([feature.lat, feature.lon], {radius: 6, color: feature.stroke, fillColor: feature.fill, fillOpacity: 0.92, weight: 2})
                      .bindPopup(feature.popup)
                      .addTo(group);
                    osmLayers.push({ feature, layer: group, visible: false });
                  });
                }
                function clearOsmLayers() {
                  while (osmLayers.length > 0) {
                    const entry = osmLayers.pop();
                    if (entry.visible) {
                      map.removeLayer(entry.layer);
                    }
                  }
                }
                function syncMatchedFeature() {
                  if (matchedMarker) {
                    map.removeLayer(matchedMarker);
                    matchedMarker = null;
                  }
                  if (matchedLine) {
                    map.removeLayer(matchedLine);
                    matchedLine = null;
                  }
                  if (!matchedFeature || matchedFeature.lat == null || matchedFeature.lon == null) {
                    return;
                  }
                  matchedMarker = L.circleMarker([matchedFeature.lat, matchedFeature.lon], {
                    radius: 9,
                    color: '#4caf50',
                    fillColor: '#66bb6a',
                    fillOpacity: 0.95,
                    weight: 3
                  }).addTo(map).bindPopup(matchedFeature.popup);
                  matchedLine = L.polyline(
                    [[currentLocation.lat, currentLocation.lon], [matchedFeature.lat, matchedFeature.lon]],
                    { color: '#4caf50', weight: 4, dashArray: '8 6' }
                  ).addTo(map);
                }
                function syncLayerVisibility(lat, lon, bundledRadius, osmRadius) {
                  ensureBundledLayers();
                  ensureOsmLayers();
                  bundledLayers.forEach((entry) => {
                    const shouldShow = true;
                    if (shouldShow && !entry.visible) {
                      entry.layer.addTo(map);
                      entry.visible = true;
                    } else if (!shouldShow && entry.visible) {
                      map.removeLayer(entry.layer);
                      entry.visible = false;
                    }
                  });
                  osmLayers.forEach((entry) => {
                    const shouldShow = distanceMeters(lat, lon, entry.feature.lat, entry.feature.lon) <= osmRadius;
                    if (shouldShow && !entry.visible) {
                      entry.layer.addTo(map);
                      entry.visible = true;
                    } else if (!shouldShow && entry.visible) {
                      map.removeLayer(entry.layer);
                      entry.visible = false;
                    }
                  });
                }
                function replaceNearbyData(nextBundledFeatures, nextOsmFeatures) {
                  bundledFeatures = nextBundledFeatures || [];
                  osmFeatures = nextOsmFeatures || [];
                  clearBundledLayers();
                  clearOsmLayers();
                  syncLayerVisibility(currentLocation.lat, currentLocation.lon, currentBundledRadius, currentOsmRadius);
                }
                window.updateCurrentLocation = function(lat, lon, osmRadius, bundledRadius) {
                  currentLocation = { lat, lon };
                  currentOsmRadius = osmRadius;
                  currentBundledRadius = bundledRadius;
                  currentMarker.setLatLng([lat, lon]);
                  osmRadiusCircle.setLatLng([lat, lon]);
                  osmRadiusCircle.setRadius(osmRadius);
                  if (matchedLine && matchedFeature) {
                    matchedLine.setLatLngs([[lat, lon], [matchedFeature.lat, matchedFeature.lon]]);
                  }
                  syncLayerVisibility(lat, lon, bundledRadius, osmRadius);
                };
                window.updateDebugMapData = function(payload) {
                  if (!payload) return;
                  if (payload.current) {
                    window.updateCurrentLocation(
                      payload.current.lat,
                      payload.current.lon,
                      payload.osmRadius || currentOsmRadius,
                      payload.bundledRadius || currentBundledRadius
                    );
                  }
                  if (payload.bundled || payload.osm) {
                    replaceNearbyData(payload.bundled || [], payload.osm || []);
                  }
                  if (Object.prototype.hasOwnProperty.call(payload, 'matched')) {
                    matchedFeature = payload.matched;
                    syncMatchedFeature();
                  }
                };
                window.recenterToCurrentLocation = function() {
                  map.setView([currentLocation.lat, currentLocation.lon], map.getZoom(), { animate: true });
                };
                map.setView([${base.currentLat}, ${base.currentLon}], 16);
                syncMatchedFeature();
                syncLayerVisibility(${base.currentLat}, ${base.currentLon}, ${currentBundleRadiusMeters()}, ${currentOsmRadiusMeters()});
              </script>
            </body>
            </html>
        """.trimIndent()
    }

    fun toErrorHtml(message: String): String {
        return """
            <!doctype html>
            <html lang="ko">
            <head>
              <meta charset="utf-8" />
              <style>
                html, body { height: 100%; margin: 0; padding: 0; background: #101418; color: #eceff1; font-family: sans-serif; }
                body { display: flex; align-items: center; justify-content: center; padding: 24px; text-align: center; }
                .box { max-width: 460px; line-height: 1.6; }
              </style>
            </head>
            <body>
              <div class="box">
                <h3>지도 세션을 불러오지 못했습니다</h3>
                <p>${escapeHtml(message)}</p>
              </div>
            </body>
            </html>
        """.trimIndent()
    }

    fun toExternalMapIntents(): List<Intent> {
        val googleUrl =
            "https://www.google.com/maps/search/?api=1&query=${base.currentLat},${base.currentLon}"
        val geoLabel = Uri.encode("VIA current location")
        val geoIntent =
            Intent(
                Intent.ACTION_VIEW,
                Uri.parse("geo:${base.currentLat},${base.currentLon}?q=${base.currentLat},${base.currentLon}($geoLabel)")
            )
        val googleMapsIntent =
            Intent(Intent.ACTION_VIEW, Uri.parse(googleUrl)).setPackage("com.google.android.apps.maps")
        val browserIntent = Intent(Intent.ACTION_VIEW, Uri.parse(googleUrl))
        val samsungInternetIntent =
            Intent(Intent.ACTION_VIEW, Uri.parse(googleUrl)).setPackage("com.sec.android.app.sbrowser")
        return listOf(googleMapsIntent, browserIntent, samsungInternetIntent, geoIntent)
    }

    companion object {
        fun fromIntent(
            intent: Intent,
            activity: AppCompatActivity
        ): DebugMapState {
            val base = resolveInitialBaseState(activity, DebugMapActivity.readIntent(intent))
            val currentPoint = GeoPoint(base.currentLat, base.currentLon)
            val nearbyFeatures =
                runCatching { loadBundledNearbyFeatures(activity, currentPoint, BUNDLED_LOAD_RADIUS_METERS) }
                    .getOrDefault(emptyList())
            return DebugMapState(base = base, nearbyFeatures = nearbyFeatures)
        }
    }

    fun withOsmCrossings(
        fetchedCrossings: List<OsmNearbyCrossing>
    ): DebugMapState {
        return copy(osmCrossings = fetchedCrossings)
    }

    fun withCurrentLocation(
        currentPoint: GeoPoint
    ): DebugMapState {
        return copy(
            base = base.copy(
                currentLat = currentPoint.latitude,
                currentLon = currentPoint.longitude
            )
        )
    }

    fun withNearbyData(
        bundledNearby: List<NearbyFeature>,
        osmNearby: List<OsmNearbyCrossing>
    ): DebugMapState {
        return copy(
            nearbyFeatures = bundledNearby,
            osmCrossings = osmNearby
        )
    }

    private fun bundledFeaturesJs(): String =
        nearbyFeatures.joinToString(prefix = "[", postfix = "]") { nearby ->
            val stroke =
                when {
                    nearby.kind.contains("ped_signal", ignoreCase = true) -> "#ff7043"
                    else -> "#ffb300"
                }
            val fill =
                when {
                    nearby.kind.contains("ped_signal", ignoreCase = true) -> "#ffab91"
                    else -> "#ffca28"
                }
            """
            {
              lat: ${nearby.lat},
              lon: ${nearby.lon},
              stroke: '$stroke',
              fill: '$fill',
              popup: ${jsString("${nearby.kind} · ${String.format(Locale.US, "%.1f", nearby.distanceMeters)}m\n${nearby.id}")}
            }
            """.trimIndent()
        }

    private fun osmFeaturesJs(): String =
        osmCrossings.joinToString(prefix = "[", postfix = "]") { crossing ->
            val stroke = if (crossing.signalControlled) "#ef5350" else "#ab47bc"
            val fill = if (crossing.signalControlled) "#ef9a9a" else "#ce93d8"
            val geometryJs =
                if (crossing.geometry.size >= 2) {
                    crossing.geometry.joinToString(prefix = "[", postfix = "]") {
                        "[${it.latitude}, ${it.longitude}]"
                    }
                } else {
                    "[]"
                }
            """
            {
              lat: ${crossing.lat},
              lon: ${crossing.lon},
              stroke: '$stroke',
              fill: '$fill',
              popup: ${jsString("${crossing.kind} · ${String.format(Locale.US, "%.1f", crossing.distanceMeters)}m\n${crossing.id}")},
              geometry: $geometryJs
            }
            """.trimIndent()
        }

    private fun matchedFeatureJs(): String =
        if (base.matchedLat != null && base.matchedLon != null) {
            """
            {
              lat: ${base.matchedLat},
              lon: ${base.matchedLon},
              popup: ${jsString("${base.matchedKind ?: "match"}\n${base.matchedId ?: ""}")}
            }
            """.trimIndent()
        } else {
            "null"
        }
}

private data class NearbyFeature(
    val id: String,
    val kind: String,
    val lat: Double,
    val lon: Double,
    val distanceMeters: Float
)

private fun loadBundledNearbyFeatures(
    activity: AppCompatActivity,
    currentPoint: GeoPoint,
    radiusMeters: Int
): List<NearbyFeature> {
    val store = MapTileStore(activity)
    val loadResult = store.loadWithinRadius(currentPoint, radiusMeters)
    return loadResult.features
        .distinctBy { it.id }
        .map {
            val distanceMeters = haversineDistanceMeters(currentPoint, it.point)
            NearbyFeature(
                id = it.id,
                kind = it.kind.wireName,
                lat = it.point.latitude,
                lon = it.point.longitude,
                distanceMeters = distanceMeters
            )
        }
        .sortedBy { it.distanceMeters }
}

private fun formatCoord(value: Double): String = String.format(Locale.US, "%.6f", value)

private fun resolveInitialBaseState(
    activity: AppCompatActivity,
    base: BaseMapDebugState
): BaseMapDebugState {
    if (!base.currentLat.isNaN() && !base.currentLon.isNaN()) {
        return base
    }
    val fallback = bestAvailableLocation(activity) ?: return base
    return base.copy(
        currentLat = fallback.latitude,
        currentLon = fallback.longitude,
        currentAccMeters = if (fallback.hasAccuracy()) fallback.accuracy else base.currentAccMeters
    )
}

private fun bestAvailableLocation(activity: AppCompatActivity): Location? {
    val hasFine =
        ContextCompat.checkSelfPermission(activity, Manifest.permission.ACCESS_FINE_LOCATION) ==
            PackageManager.PERMISSION_GRANTED
    val hasCoarse =
        ContextCompat.checkSelfPermission(activity, Manifest.permission.ACCESS_COARSE_LOCATION) ==
            PackageManager.PERMISSION_GRANTED
    if (!hasFine && !hasCoarse) {
        return null
    }
    val locationManager = activity.getSystemService(LocationManager::class.java)
    val candidates =
        buildList {
            runCatching {
                if (locationManager?.isProviderEnabled(LocationManager.GPS_PROVIDER) == true) {
                    locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER)?.let { add(it) }
                }
            }
            runCatching {
                if (locationManager?.isProviderEnabled(LocationManager.NETWORK_PROVIDER) == true) {
                    locationManager.getLastKnownLocation(LocationManager.NETWORK_PROVIDER)?.let { add(it) }
                }
            }
        }
    return candidates.minWithOrNull(
        compareBy<Location> { if (it.hasAccuracy()) it.accuracy else Float.MAX_VALUE }
            .thenByDescending { it.time }
    )
}

private fun escapeHtml(value: String): String =
    value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

private fun jsString(value: String): String {
    return buildString {
        append('"')
        value.forEach { ch ->
            when (ch) {
                '\\' -> append("\\\\")
                '"' -> append("\\\"")
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                else -> append(ch)
            }
        }
        append('"')
    }
}

private fun shortId(value: String): String {
    return if (value.length <= 28) value else value.take(28) + "…"
}

private fun blankTileResponse(): WebResourceResponse {
    return WebResourceResponse(
        "image/png",
        "binary",
        200,
        "OK",
        mapOf("Cache-Control" to "no-store"),
        ByteArrayInputStream(TRANSPARENT_PNG_1X1)
    )
}

private fun DebugMapActivity.readCachedTile(url: String): WebResourceResponse? {
    val baseFile = tileCacheFile(url)
    val metaFile = File(baseFile.parentFile, "${baseFile.name}.meta")
    if (!baseFile.exists() || !metaFile.exists()) {
        return null
    }
    return runCatching {
        val meta = metaFile.readText(StandardCharsets.UTF_8).split("\n")
        val expiryEpochMs = meta.getOrNull(0)?.toLongOrNull() ?: 0L
        val mimeType = meta.getOrNull(1)?.ifBlank { "image/png" } ?: "image/png"
        val encoding = meta.getOrNull(2)?.ifBlank { "binary" } ?: "binary"
        if (expiryEpochMs in 1 until System.currentTimeMillis()) {
            baseFile.delete()
            metaFile.delete()
            return null
        }
        val bytes = baseFile.readBytes()
        WebResourceResponse(
            mimeType,
            encoding,
            200,
            "OK",
            mapOf("Cache-Control" to "private"),
            ByteArrayInputStream(bytes)
        )
    }.getOrNull()
}

private fun DebugMapActivity.writeCachedTile(
    url: String,
    payload: ByteArray,
    mimeType: String,
    encoding: String,
    expiryEpochMs: Long
) {
    runCatching {
        val baseFile = tileCacheFile(url)
        val metaFile = File(baseFile.parentFile, "${baseFile.name}.meta")
        baseFile.parentFile?.mkdirs()
        baseFile.writeBytes(payload)
        metaFile.writeText(
            listOf(expiryEpochMs.toString(), mimeType, encoding).joinToString("\n"),
            StandardCharsets.UTF_8
        )
    }
}

private fun parseExpiryEpochMs(connection: HttpURLConnection): Long {
    val cacheControl = connection.getHeaderField("Cache-Control").orEmpty()
    val maxAge =
        Regex("""max-age=(\d+)""")
            .find(cacheControl)
            ?.groupValues
            ?.getOrNull(1)
            ?.toLongOrNull()
    return if (maxAge != null) {
        System.currentTimeMillis() + maxAge * 1000L
    } else {
        System.currentTimeMillis() + 10 * 60 * 1000L
    }
}

private fun DebugMapActivity.tileCacheFile(url: String): File {
    val hash =
        MessageDigest.getInstance("SHA-256")
            .digest(url.toByteArray(StandardCharsets.UTF_8))
            .joinToString("") { "%02x".format(it) }
    return MapDebugCacheManager.tileCacheDir(this).resolve("$hash.bin")
}

private val TRANSPARENT_PNG_1X1 = byteArrayOf(
    0x89.toByte(), 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
    0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
    0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4.toByte(),
    0x89.toByte(), 0x00, 0x00, 0x00, 0x0D, 0x49, 0x44, 0x41,
    0x54, 0x78, 0x9C.toByte(), 0x63, 0x00, 0x01, 0x00, 0x00,
    0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4.toByte(), 0x00,
    0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE.toByte(),
    0x42, 0x60, 0x82.toByte()
)
