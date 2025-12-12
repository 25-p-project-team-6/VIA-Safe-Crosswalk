package kr.co.gachon.pproject6.via.util

import java.util.ArrayDeque

class PerformanceTracker {
    
    // Member variables for stats
    private var lastFpsTimestamp = System.currentTimeMillis()
    private var frameCount = 0

    // Store timestamps and latencies for 10s sliding window
    // Pair(timestamp, latency)
    private val frameData = ArrayDeque<Pair<Long, Long>>()
    
    // Calculated stats
    var currentFpsStr: String = "FPS: 0.00"
        private set
    var avgFpsStr: String = "Avg FPS: 0.00"
        private set
    var avgLatencyStr: String = "Avg Latency: 0ms"
        private set

    fun clear() {
        frameData.clear()
        currentFpsStr = "FPS: 0"
        avgFpsStr = "Avg FPS: 0"
        avgLatencyStr = "Avg Latency: 0ms"
        frameCount = 0
        lastFpsTimestamp = System.currentTimeMillis()
    }

    fun update(inferenceTime: Long) {
        val currentTime = System.currentTimeMillis()

        // Add current frame data
        frameData.addLast(Pair(currentTime, inferenceTime))

        // Remove old data (older than 10 seconds)
        while (!frameData.isEmpty() && currentTime - frameData.peekFirst().first > 10000) {
            frameData.removeFirst()
        }

        // Calculate Average Latency (10s window)
        if (!frameData.isEmpty()) {
            var totalLatency = 0L
            for (p in frameData) {
                totalLatency += p.second
            }
            val avgLatency = totalLatency / frameData.size
            avgLatencyStr = "Avg Latency: ${avgLatency}ms"
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
                val oldestTime = frameData.peekFirst()?.first
                val duration = currentTime - oldestTime!!

                if (duration > 0) {
                    val calculatedAvgFps =
                        frameData.size * 1000.0 / (if (duration < 100) 1000.0 else duration.toDouble())
                    avgFpsStr = String.format("Avg FPS: %.2f", calculatedAvgFps)
                }
            }
        }
    }
}
