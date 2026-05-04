package kr.co.gachon.pproject6.via.context

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Looper
import androidx.core.content.ContextCompat
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.sqrt
import java.util.Locale

class CrossingSupportManager(
    context: Context,
    private val config: CrossingSupportConfig = CrossingSupportConfig(),
    private val timeProvider: () -> Long = System::currentTimeMillis
) : SensorEventListener {
    private val appContext = context.applicationContext
    private val sensorManager =
        appContext.getSystemService(Context.SENSOR_SERVICE) as? SensorManager
    private val gyroscope = sensorManager?.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val gravitySensor = sensorManager?.getDefaultSensor(Sensor.TYPE_GRAVITY)
    private val locationManager =
        appContext.getSystemService(Context.LOCATION_SERVICE) as? LocationManager

    private var started = false
    private var lastMotionAt: Long = Long.MIN_VALUE
    private var lookingDownStartedAt: Long = Long.MIN_VALUE
    private var lookingUpStartedAt: Long = Long.MIN_VALUE
    private var currentTiltDegrees: Float = 0f
    private var currentSignedTiltDegrees: Float = 0f
    private var lastLocationMovementAt: Long = Long.MIN_VALUE
    private var lastLocationSample: Location? = null
    private var crossingWindowActive = false
    private var crossingWindowStartedAt: Long = Long.MIN_VALUE
    private var crossingWindowDistanceMeters: Float = 0f
    private var lastCrossingWindowLocation: Location? = null
    private var gpsRegistered = false
    private var networkRegistered = false
    private var lastHeadingDegrees: Float? = null
    private var lookingDownRawTiltRangeStartDegrees = config.lookingDownRawTiltRangeStartDegrees
    private var lookingDownRawTiltRangeEndDegrees = config.lookingDownRawTiltRangeEndDegrees
    private var lookingUpRawTiltRangeStartDegrees = config.lookingUpRawTiltRangeStartDegrees
    private var lookingUpRawTiltRangeEndDegrees = config.lookingUpRawTiltRangeEndDegrees
    private val mapProximityManager = MapProximityManager(appContext, timeProvider)

    private val locationListener = LocationListener { location ->
        handleLocation(location)
    }

    fun start() {
        if (started) {
            refreshLocationRegistration()
            return
        }
        started = true
        gyroscope?.let {
            sensorManager?.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        gravitySensor?.let {
            sensorManager?.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        refreshLocationRegistration()
    }

    fun stop() {
        if (!started) {
            return
        }
        started = false
        sensorManager?.unregisterListener(this)
        unregisterLocationUpdates()
    }

    fun updateLookingDownThresholdDegrees(value: Float) {
        val clampedThresholdDegrees = value.coerceIn(70f, 140f)
        lookingDownRawTiltRangeEndDegrees = -clampedThresholdDegrees
    }

    fun currentLookingDownThresholdDegrees(): Float = abs(lookingDownRawTiltRangeEndDegrees)

    fun updateLookingUpThresholdDegrees(value: Float) {
        val clampedThresholdDegrees = value.coerceIn(90f, 170f)
        lookingUpRawTiltRangeEndDegrees = clampedThresholdDegrees
    }

    fun currentLookingUpThresholdDegrees(): Float = lookingUpRawTiltRangeEndDegrees

    fun setCrossingWindowActive(isActive: Boolean) {
        val now = timeProvider()
        if (isActive) {
            if (!crossingWindowActive) {
                crossingWindowActive = true
                crossingWindowStartedAt = now
                crossingWindowDistanceMeters = 0f
                lastCrossingWindowLocation = lastLocationSample
            }
        } else if (crossingWindowActive) {
            clearCrossingWindow()
        }
    }

    @SuppressLint("MissingPermission")
    fun refreshLocationRegistration() {
        if (!started) {
            return
        }
        unregisterLocationUpdates()
        if (!hasLocationPermission()) {
            return
        }

        val manager = locationManager ?: return
        val looper = Looper.getMainLooper()
        seedLastKnownLocation(manager)

        if (manager.isProviderEnabledSafe(LocationManager.GPS_PROVIDER)) {
            manager.requestLocationUpdates(
                LocationManager.GPS_PROVIDER,
                config.locationMinUpdateIntervalMs,
                config.locationMinDistanceMeters,
                locationListener,
                looper
            )
            gpsRegistered = true
        }
        if (manager.isProviderEnabledSafe(LocationManager.NETWORK_PROVIDER)) {
            manager.requestLocationUpdates(
                LocationManager.NETWORK_PROVIDER,
                config.locationMinUpdateIntervalMs,
                config.locationMinDistanceMeters,
                locationListener,
                looper
            )
            networkRegistered = true
        }
    }

    @SuppressLint("MissingPermission")
    private fun seedLastKnownLocation(
        manager: LocationManager
    ) {
        val candidates =
            buildList {
                if (manager.isProviderEnabledSafe(LocationManager.GPS_PROVIDER)) {
                    manager.getLastKnownLocationSafe(LocationManager.GPS_PROVIDER)?.let { add(it) }
                }
                if (manager.isProviderEnabledSafe(LocationManager.NETWORK_PROVIDER)) {
                    manager.getLastKnownLocationSafe(LocationManager.NETWORK_PROVIDER)?.let { add(it) }
                }
            }
        val best =
            candidates.minWithOrNull(
                compareBy<Location> {
                    if (it.hasAccuracy()) it.accuracy else Float.MAX_VALUE
                }.thenByDescending { it.time }
            )
        if (best != null) {
            handleLocation(best)
        }
    }

    fun snapshot(): CrossingSupportSnapshot {
        val now = timeProvider()
        val hasRecentGyroMotion =
            lastMotionAt != Long.MIN_VALUE && now - lastMotionAt <= config.motionHoldMs
        val isLookingDown =
            lookingDownStartedAt != Long.MIN_VALUE &&
                now - lookingDownStartedAt >= config.lookingDownHoldMs
        val isLookingUp =
            lookingUpStartedAt != Long.MIN_VALUE &&
                now - lookingUpStartedAt >= config.lookingUpHoldMs
        val hasRecentLocationMovement =
            lastLocationMovementAt != Long.MIN_VALUE &&
                now - lastLocationMovementAt <= config.locationHoldMs
        val crossingWindowElapsedMs =
            if (crossingWindowActive && crossingWindowStartedAt != Long.MIN_VALUE) {
                now - crossingWindowStartedAt
            } else {
                0L
            }
        return CrossingSupportSnapshot(
            isCrossingWindowActive = crossingWindowActive,
            hasRecentGyroMotion = hasRecentGyroMotion,
            isLookingDown = isLookingDown,
            isLookingUp = isLookingUp,
            currentTiltDegrees = currentTiltDegrees,
            currentSignedTiltDegrees = currentSignedTiltDegrees,
            hasRecentLocationMovement = hasRecentLocationMovement,
            crossingWindowDistanceMeters = crossingWindowDistanceMeters,
            crossingWindowElapsedMs = crossingWindowElapsedMs,
            nextCrosswalkDistanceThresholdMeters = config.nextCrosswalkDistanceThresholdMeters,
            nextCrosswalkMinActiveMs = config.nextCrosswalkMinActiveMs,
            mapProximitySnapshot = mapProximityManager.snapshot(),
            currentLocationLatitude = lastLocationSample?.latitude,
            currentLocationLongitude = lastLocationSample?.longitude,
            currentLocationAccuracyMeters =
                if (lastLocationSample?.hasAccuracy() == true) lastLocationSample?.accuracy else null,
            currentHeadingDegrees = lastHeadingDegrees
        )
    }

    fun reset() {
        lastMotionAt = Long.MIN_VALUE
        lookingDownStartedAt = Long.MIN_VALUE
        lookingUpStartedAt = Long.MIN_VALUE
        currentTiltDegrees = 0f
        currentSignedTiltDegrees = 0f
        lastLocationMovementAt = Long.MIN_VALUE
        lastLocationSample = null
        lastHeadingDegrees = null
        clearCrossingWindow()
        mapProximityManager.reset()
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_GYROSCOPE -> {
                val magnitude = sqrt(
                    event.values[0] * event.values[0] +
                        event.values[1] * event.values[1] +
                        event.values[2] * event.values[2]
                )
                if (magnitude >= config.gyroMotionThresholdRadPerSec) {
                    lastMotionAt = timeProvider()
                }
            }
            Sensor.TYPE_GRAVITY -> {
                val signedTiltDegrees =
                    calculateSignedTiltFromUprightDegrees(
                        event.values[0],
                        event.values[1],
                        event.values[2]
                    )
                val tiltDegrees = abs(signedTiltDegrees)
                currentSignedTiltDegrees = signedTiltDegrees
                currentTiltDegrees = tiltDegrees
                if (signedTiltDegrees in lookingDownRawTiltRangeStartDegrees..lookingDownRawTiltRangeEndDegrees) {
                    if (lookingDownStartedAt == Long.MIN_VALUE) {
                        lookingDownStartedAt = timeProvider()
                    }
                } else {
                    lookingDownStartedAt = Long.MIN_VALUE
                }

                if (signedTiltDegrees in lookingUpRawTiltRangeStartDegrees..lookingUpRawTiltRangeEndDegrees) {
                    if (lookingUpStartedAt == Long.MIN_VALUE) {
                        lookingUpStartedAt = timeProvider()
                    }
                } else {
                    lookingUpStartedAt = Long.MIN_VALUE
                }
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    private fun handleLocation(location: Location) {
        val now = timeProvider()
        val previous = lastLocationSample
        val speedMps =
            when {
                location.hasSpeed() -> location.speed
                previous != null -> {
                    val dtMillis = (location.time - previous.time).coerceAtLeast(1L)
                    previous.distanceTo(location) / (dtMillis / 1000f)
                }
                else -> 0f
            }
        val distanceMeters = previous?.distanceTo(location) ?: 0f

        if (speedMps >= config.locationSpeedThresholdMps ||
            distanceMeters >= config.locationDistanceThresholdMeters
        ) {
            lastLocationMovementAt = now
        }

        if (crossingWindowActive) {
            val lastCrossing = lastCrossingWindowLocation
            if (lastCrossing != null) {
                val segmentDistance = lastCrossing.distanceTo(location)
                if (segmentDistance in config.crossingDistanceSegmentRangeMeters) {
                    crossingWindowDistanceMeters += segmentDistance
                }
            }
            lastCrossingWindowLocation = location
        }

        val headingDegrees =
            when {
                location.hasBearing() -> normalizeBearingDegrees(location.bearing)
                previous != null && distanceMeters >= config.locationDistanceThresholdMeters ->
                    normalizeBearingDegrees(previous.bearingTo(location))
                else -> lastHeadingDegrees
            }
        lastHeadingDegrees = headingDegrees
        lastLocationSample = location
        mapProximityManager.onLocation(location, headingDegrees)
    }

    private fun clearCrossingWindow() {
        crossingWindowActive = false
        crossingWindowStartedAt = Long.MIN_VALUE
        crossingWindowDistanceMeters = 0f
        lastCrossingWindowLocation = null
    }

    private fun unregisterLocationUpdates() {
        val manager = locationManager ?: return
        if (gpsRegistered || networkRegistered) {
            manager.removeUpdates(locationListener)
        }
        gpsRegistered = false
        networkRegistered = false
    }

    private fun hasLocationPermission(): Boolean {
        val fine =
            ContextCompat.checkSelfPermission(
                appContext,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED
        val coarse =
            ContextCompat.checkSelfPermission(
                appContext,
                Manifest.permission.ACCESS_COARSE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED
        return fine || coarse
    }

    private fun LocationManager.isProviderEnabledSafe(provider: String): Boolean {
        return runCatching { isProviderEnabled(provider) }.getOrDefault(false)
    }

    @SuppressLint("MissingPermission")
    private fun LocationManager.getLastKnownLocationSafe(provider: String): Location? {
        return runCatching { getLastKnownLocation(provider) }.getOrNull()
    }
}

internal fun calculateTiltFromUprightDegrees(x: Float, y: Float, z: Float): Float {
    return abs(calculateSignedTiltFromUprightDegrees(x, y, z))
}

internal fun calculateSignedTiltFromUprightDegrees(x: Float, y: Float, z: Float): Float {
    if (y == 0f && z == 0f) {
        return 0f
    }
    return Math.toDegrees(atan2((-z).toDouble(), (-y).toDouble())).toFloat()
}

data class CrossingSupportConfig(
    val gyroMotionThresholdRadPerSec: Float = 0.8f,
    val motionHoldMs: Long = 2_500L,
    val lookingDownRawTiltRangeStartDegrees: Float = -160f,
    val lookingDownRawTiltRangeEndDegrees: Float = -90f,
    val lookingDownHoldMs: Long = 900L,
    val lookingUpRawTiltRangeStartDegrees: Float = 90f,
    val lookingUpRawTiltRangeEndDegrees: Float = 120f,
    val lookingUpHoldMs: Long = 900L,
    val locationSpeedThresholdMps: Float = 0.7f,
    val locationDistanceThresholdMeters: Float = 2.0f,
    val locationHoldMs: Long = 4_000L,
    val nextCrosswalkDistanceThresholdMeters: Float = 8.0f,
    val nextCrosswalkMinActiveMs: Long = 6_000L,
    val crossingDistanceSegmentRangeMeters: ClosedFloatingPointRange<Float> = 0.5f..15.0f,
    val locationMinUpdateIntervalMs: Long = 1_000L,
    val locationMinDistanceMeters: Float = 0.5f
)

data class CrossingSupportSnapshot(
    val isCrossingWindowActive: Boolean = false,
    val hasRecentGyroMotion: Boolean = false,
    val isLookingDown: Boolean = false,
    val isLookingUp: Boolean = false,
    val currentTiltDegrees: Float = 0f,
    val currentSignedTiltDegrees: Float = 0f,
    val hasRecentLocationMovement: Boolean = false,
    val crossingWindowDistanceMeters: Float = 0f,
    val crossingWindowElapsedMs: Long = 0L,
    val nextCrosswalkDistanceThresholdMeters: Float = 8.0f,
    val nextCrosswalkMinActiveMs: Long = 6_000L,
    val mapProximitySnapshot: MapProximitySnapshot = MapProximitySnapshot(),
    val currentLocationLatitude: Double? = null,
    val currentLocationLongitude: Double? = null,
    val currentLocationAccuracyMeters: Float? = null,
    val currentHeadingDegrees: Float? = null
) {
    val supportsWalkContinuation: Boolean
        get() = hasRecentGyroMotion || hasRecentLocationMovement || isLookingDown

    fun toDebugSummary(): String {
        val latSummary = currentLocationLatitude?.let { String.format(Locale.US, "%.6f", it) } ?: "n/a"
        val lonSummary = currentLocationLongitude?.let { String.format(Locale.US, "%.6f", it) } ?: "n/a"
        val accSummary = currentLocationAccuracyMeters?.let { String.format(Locale.US, "%.1f", it) } ?: "n/a"
        val headingSummary = currentHeadingDegrees?.let { String.format(Locale.US, "%.0f", it) } ?: "n/a"
        return "motion=$hasRecentGyroMotion, down=$isLookingDown, up=$isLookingUp, tilt=${String.format(Locale.US, "%.0f", currentTiltDegrees)}, signedTilt=${String.format(Locale.US, "%.0f", currentSignedTiltDegrees)}, gps=$hasRecentLocationMovement, keep=$supportsWalkContinuation, window=$isCrossingWindowActive, dist=${String.format(Locale.US, "%.1f", crossingWindowDistanceMeters)}, elapsed=${crossingWindowElapsedMs}ms, lat=$latSummary, lon=$lonSummary, acc=${accSummary}m, heading=$headingSummary, ${mapProximitySnapshot.toDebugSummary()}"
    }
}
