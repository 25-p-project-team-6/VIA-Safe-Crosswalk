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
import kotlin.math.sqrt

class CrossingSupportManager(
    context: Context,
    private val config: CrossingSupportConfig = CrossingSupportConfig(),
    private val timeProvider: () -> Long = System::currentTimeMillis
) : SensorEventListener {
    private val appContext = context.applicationContext
    private val sensorManager =
        appContext.getSystemService(Context.SENSOR_SERVICE) as? SensorManager
    private val gyroscope = sensorManager?.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val locationManager =
        appContext.getSystemService(Context.LOCATION_SERVICE) as? LocationManager

    private var started = false
    private var lastMotionAt: Long = Long.MIN_VALUE
    private var lastLocationMovementAt: Long = Long.MIN_VALUE
    private var lastLocationSample: Location? = null
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
        val hasRecentLocationMovement =
            lastLocationMovementAt != Long.MIN_VALUE &&
                now - lastLocationMovementAt <= config.locationHoldMs
        return CrossingSupportSnapshot(
            hasRecentGyroMotion = hasRecentGyroMotion,
            hasRecentLocationMovement = hasRecentLocationMovement
        )
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type != Sensor.TYPE_GYROSCOPE) {
            return
        }
        val magnitude = sqrt(
            event.values[0] * event.values[0] +
                event.values[1] * event.values[1] +
                event.values[2] * event.values[2]
        )
        if (magnitude >= config.gyroMotionThresholdRadPerSec) {
            lastMotionAt = timeProvider()
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

data class CrossingSupportConfig(
    val gyroMotionThresholdRadPerSec: Float = 0.8f,
    val motionHoldMs: Long = 2_500L,
    val locationSpeedThresholdMps: Float = 0.7f,
    val locationDistanceThresholdMeters: Float = 2.0f,
    val locationHoldMs: Long = 4_000L,
    val locationMinUpdateIntervalMs: Long = 1_000L,
    val locationMinDistanceMeters: Float = 0.5f
)

data class CrossingSupportSnapshot(
    val hasRecentGyroMotion: Boolean = false,
    val hasRecentLocationMovement: Boolean = false
) {
    val supportsWalkContinuation: Boolean
        get() = hasRecentGyroMotion || hasRecentLocationMovement

    fun toDebugSummary(): String {
        return "motion=$hasRecentGyroMotion, gps=$hasRecentLocationMovement, keep=$supportsWalkContinuation"
    }
}
