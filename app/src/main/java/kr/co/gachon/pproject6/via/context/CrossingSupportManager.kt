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
import kr.co.gachon.pproject6.via.ml.UserGuidanceState
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
    private var lookingDownTiltThresholdDegrees = config.lookingDownTiltThresholdDegrees
    private var currentTiltDegrees: Float = 0f
    private var lastLocationMovementAt: Long = Long.MIN_VALUE
    private var lastLocationSample: Location? = null
    private var crossingActive = false
    private var crossingStartedAt: Long = Long.MIN_VALUE
    private var distanceSinceCrossingStartMeters: Float = 0f
    private var lastCrossingLocation: Location? = null
    private var gpsRegistered = false
    private var networkRegistered = false

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
        lookingDownTiltThresholdDegrees = value
    }

    fun currentLookingDownThresholdDegrees(): Float = lookingDownTiltThresholdDegrees

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

    fun snapshot(): CrossingSupportSnapshot {
        val now = timeProvider()
        val hasRecentGyroMotion =
            lastMotionAt != Long.MIN_VALUE && now - lastMotionAt <= config.motionHoldMs
        val isLookingDown =
            lookingDownStartedAt != Long.MIN_VALUE &&
                now - lookingDownStartedAt >= config.lookingDownHoldMs
        val hasRecentLocationMovement =
            lastLocationMovementAt != Long.MIN_VALUE &&
                now - lastLocationMovementAt <= config.locationHoldMs
        val crossingActiveDurationMs =
            if (crossingActive && crossingStartedAt != Long.MIN_VALUE) {
                now - crossingStartedAt
            } else {
                0L
            }
        return CrossingSupportSnapshot(
            isCrossingActive = crossingActive,
            hasRecentGyroMotion = hasRecentGyroMotion,
            isLookingDown = isLookingDown,
            currentTiltDegrees = currentTiltDegrees,
            hasRecentLocationMovement = hasRecentLocationMovement,
            distanceSinceCrossingStartMeters = distanceSinceCrossingStartMeters,
            crossingActiveDurationMs = crossingActiveDurationMs,
            nextCrosswalkDistanceThresholdMeters = config.nextCrosswalkDistanceThresholdMeters,
            nextCrosswalkMinActiveMs = config.nextCrosswalkMinActiveMs
        )
    }

    fun onGuidanceStateChanged(state: UserGuidanceState) {
        val now = timeProvider()
        if (state == UserGuidanceState.GO) {
            if (!crossingActive) {
                crossingActive = true
                crossingStartedAt = now
                distanceSinceCrossingStartMeters = 0f
                lastCrossingLocation = lastLocationSample
            }
        } else if (crossingActive) {
            crossingActive = false
            crossingStartedAt = Long.MIN_VALUE
            distanceSinceCrossingStartMeters = 0f
            lastCrossingLocation = null
        }
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
                val tiltDegrees = calculateTiltFromUprightDegrees(event.values[0], event.values[1], event.values[2])
                currentTiltDegrees = tiltDegrees
                if (tiltDegrees >= lookingDownTiltThresholdDegrees) {
                    if (lookingDownStartedAt == Long.MIN_VALUE) {
                        lookingDownStartedAt = timeProvider()
                    }
                } else {
                    lookingDownStartedAt = Long.MIN_VALUE
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

        if (crossingActive) {
            val lastCrossing = lastCrossingLocation
            if (lastCrossing != null) {
                val segmentDistance = lastCrossing.distanceTo(location)
                if (segmentDistance in config.crossingDistanceSegmentRangeMeters) {
                    distanceSinceCrossingStartMeters += segmentDistance
                }
            }
            lastCrossingLocation = location
        }

        lastLocationSample = location
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
}

internal fun calculateTiltFromUprightDegrees(x: Float, y: Float, z: Float): Float {
    if (y == 0f && z == 0f) {
        return 0f
    }
    return Math.toDegrees(atan2(abs(z).toDouble(), abs(y).toDouble())).toFloat()
}

data class CrossingSupportConfig(
    val gyroMotionThresholdRadPerSec: Float = 0.8f,
    val motionHoldMs: Long = 2_500L,
    val lookingDownTiltThresholdDegrees: Float = 45f,
    val lookingDownHoldMs: Long = 900L,
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
    val isCrossingActive: Boolean = false,
    val hasRecentGyroMotion: Boolean = false,
    val isLookingDown: Boolean = false,
    val currentTiltDegrees: Float = 0f,
    val hasRecentLocationMovement: Boolean = false,
    val distanceSinceCrossingStartMeters: Float = 0f,
    val crossingActiveDurationMs: Long = 0L,
    val nextCrosswalkDistanceThresholdMeters: Float = 8.0f,
    val nextCrosswalkMinActiveMs: Long = 6_000L
) {
    val supportsWalkContinuation: Boolean
        get() = hasRecentGyroMotion || hasRecentLocationMovement || isLookingDown

    val supportsNextCrosswalkTransition: Boolean
        get() = isCrossingActive &&
            hasRecentLocationMovement &&
            distanceSinceCrossingStartMeters >= nextCrosswalkDistanceThresholdMeters &&
            crossingActiveDurationMs >= nextCrosswalkMinActiveMs

    fun toDebugSummary(): String {
        return "motion=$hasRecentGyroMotion, down=$isLookingDown, tilt=${String.format(Locale.US, "%.0f", currentTiltDegrees)}, gps=$hasRecentLocationMovement, keep=$supportsWalkContinuation, next=$supportsNextCrosswalkTransition, dist=${String.format(Locale.US, "%.1f", distanceSinceCrossingStartMeters)}"
    }
}
