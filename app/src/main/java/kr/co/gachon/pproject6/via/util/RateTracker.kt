package kr.co.gachon.pproject6.via.util

class RateTracker(
    private val label: String,
    private val timeProvider: () -> Long = System::currentTimeMillis,
    private val sampleWindowMs: Long = 1_000L
) {
    private var windowStartedAt = timeProvider()
    private var sampleCount = 0

    var rateStr: String = "$label: 0.00"
        private set

    fun mark() {
        sampleCount++
        val currentTime = timeProvider()
        val elapsed = currentTime - windowStartedAt
        if (elapsed >= sampleWindowMs) {
            val rate = sampleCount * 1000.0 / elapsed
            rateStr = String.format("%s: %.2f", label, rate)
            sampleCount = 0
            windowStartedAt = currentTime
        }
    }

    fun clear() {
        sampleCount = 0
        windowStartedAt = timeProvider()
        rateStr = "$label: 0"
    }
}
