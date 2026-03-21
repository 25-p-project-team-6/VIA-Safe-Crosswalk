package kr.co.gachon.pproject6.via.util

import java.util.ArrayDeque

class PerformanceTracker(
    private val timeProvider: () -> Long = System::currentTimeMillis
) {
    private data class FrameMeasurement(
        val timestamp: Long,
        val inferenceTime: Long,
        val stageDurationsMs: Map<String, Long>
    )

    companion object {
        private const val WINDOW_MS = 10_000L
        private val DEFAULT_STAGE_ORDER = listOf("copy", "rotate", "detect", "analyze", "ui")
    }
    
    // Member variables for stats
    private var lastFpsTimestamp = timeProvider()
    private var frameCount = 0

    // Store timestamps and latencies for 10s sliding window.
    private val frameData = ArrayDeque<FrameMeasurement>()
    
    // Calculated stats
    var currentFpsStr: String = "FPS: 0.00"
        private set
    var avgFpsStr: String = "Avg FPS: 0.00"
        private set
    var avgLatencyStr: String = "Avg Latency: 0ms"
        private set
    var stageBreakdownStr: String = "Stages: n/a"
        private set

    fun clear() {
        frameData.clear()
        currentFpsStr = "FPS: 0"
        avgFpsStr = "Avg FPS: 0"
        avgLatencyStr = "Avg Latency: 0ms"
        stageBreakdownStr = "Stages: n/a"
        frameCount = 0
        lastFpsTimestamp = timeProvider()
    }

    fun update(inferenceTime: Long, stageDurationsMs: Map<String, Long> = emptyMap()) {
        val currentTime = timeProvider()

        // Add current frame data
        frameData.addLast(
            FrameMeasurement(
                timestamp = currentTime,
                inferenceTime = inferenceTime,
                stageDurationsMs = stageDurationsMs.toMap()
            )
        )

        // Remove old data (older than 10 seconds)
        while (true) {
            val oldestFrame = frameData.peekFirst() ?: break
            if (currentTime - oldestFrame.timestamp <= WINDOW_MS) {
                break
            }
            frameData.removeFirst()
        }

        // Calculate Average Latency (10s window)
        if (!frameData.isEmpty()) {
            var totalLatency = 0L
            val stageTotals = linkedMapOf<String, Long>()

            for (frame in frameData) {
                totalLatency += frame.inferenceTime
                for ((stage, durationMs) in frame.stageDurationsMs) {
                    stageTotals[stage] = (stageTotals[stage] ?: 0L) + durationMs
                }
            }
            val avgLatency = totalLatency / frameData.size
            avgLatencyStr = "Avg Latency: ${avgLatency}ms"

            if (stageTotals.isNotEmpty()) {
                val stageOrder = LinkedHashSet<String>().apply {
                    addAll(DEFAULT_STAGE_ORDER)
                    addAll(stageTotals.keys.sorted())
                }
                val breakdown = stageOrder
                    .mapNotNull { stage ->
                        val total = stageTotals[stage] ?: return@mapNotNull null
                        "$stage ${total / frameData.size}ms"
                    }
                    .joinToString(" | ")
                stageBreakdownStr = "Stages: $breakdown"
            } else {
                stageBreakdownStr = "Stages: n/a"
            }
        }

        frameCount++
        val timeDiff = currentTime - lastFpsTimestamp

        // Update Instant FPS every 1 second (stats for last 1 sec)
        if (timeDiff >= 1000) {
            val fps = frameCount * 1000.0 / timeDiff
            currentFpsStr = String.format("FPS: %.2f", fps)
            frameCount = 0
            lastFpsTimestamp = currentTime

            // Calculate Average FPS (10s window)
            if (!frameData.isEmpty()) {
                val oldestFrame = frameData.peekFirst() ?: return
                val duration = currentTime - oldestFrame.timestamp

                if (duration > 0) {
                    val calculatedAvgFps =
                        frameData.size * 1000.0 / (if (duration < 100) 1000.0 else duration.toDouble())
                    avgFpsStr = String.format("Avg FPS: %.2f", calculatedAvgFps)
                }
            }
        }
    }
}
