package kr.co.gachon.pproject6.via.util

import java.util.Locale

class RateTracker(
    private val label: String,
    private val timeProvider: () -> Long = System::currentTimeMillis,
    private val sampleWindowMs: Long = 1_000L
) {
    private var windowStartedAt = timeProvider()
    private var sampleCount = 0
    private var currentRate = 0.0

    var rateStr: String = formatRate(label, currentRate)
        private set

    fun mark() {
        sampleCount++
        val currentTime = timeProvider()
        val elapsed = currentTime - windowStartedAt
        if (elapsed >= sampleWindowMs) {
            currentRate = sampleCount * 1000.0 / elapsed
            rateStr = formatRate(label, currentRate)
            sampleCount = 0
            windowStartedAt = currentTime
        }
    }

    fun displayString(labelOverride: String = label): String =
        formatRate(labelOverride, currentRate)

    fun clear() {
        sampleCount = 0
        windowStartedAt = timeProvider()
        currentRate = 0.0
        rateStr = formatRate(label, currentRate)
    }

    private companion object {
        fun formatRate(label: String, rate: Double): String =
            String.format(Locale.US, "%s: %.2f", label, rate)
    }
}
