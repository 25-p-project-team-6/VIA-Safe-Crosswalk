package kr.co.gachon.pproject6.via

import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log

object PostProcessor {
    private const val TAG = "PostProcessor"

    enum class TrafficLightState {
        RED, GREEN, UNKNOWN
    }


    // State variables for robustness
    private var lastKnownState: TrafficLightState = TrafficLightState.UNKNOWN
    private var lastStateTimeTime: Long = 0
    private var consecutiveCount = 0
    private var candidateState: TrafficLightState = TrafficLightState.UNKNOWN

    // Constants
    private const val PERSISTENCE_DURATION_MS = 5000L // Keep state for 5 seconds even if lost
    private const val TRIGGER_THRESHOLD = 3 // Need 3 consecutive frames to switch
    fun applyColorCorrection(
        bitmap: Bitmap,
        boxes: List<OverlayView.BoundingBox>
    ): List<OverlayView.BoundingBox> {
        val correctedBoxes = mutableListOf<OverlayView.BoundingBox>()

        for (box in boxes) {
            var clsName = box.clsName
            var newClsName = clsName
            val newScore = box.score // Keep confidence score
            var debugRatio = -1f

            if (clsName.equals("green", ignoreCase = true) || clsName.equals(
                    "red",
                    ignoreCase = true
                )
            ) {

                // Crop ROI
                val rect = box.box
                val x = (rect.left * bitmap.width).toInt().coerceIn(0, bitmap.width - 1)
                val y = (rect.top * bitmap.height).toInt().coerceIn(0, bitmap.height - 1)
                val w = (rect.width() * bitmap.width).toInt().coerceIn(1, bitmap.width - x)
                val h = (rect.height() * bitmap.height).toInt().coerceIn(1, bitmap.height - y)

                if (w > 0 && h > 0) {
                    val pixels = IntArray(w * h)
                    bitmap.getPixels(pixels, 0, w, x, y, w, h)

                    val isCurrentGreen = clsName.equals("green", ignoreCase = true)
                    // Calculate ratio for current color
                    val currentRatio = calculateColorRatio(pixels, isCurrentGreen)
                    debugRatio = currentRatio

                    if (currentRatio <= 0.05f) {
                        // Check other color
                        val otherRatio = calculateColorRatio(pixels, !isCurrentGreen)

                        if (otherRatio > 0.05f) {
                            // Swap!
                            newClsName = if (isCurrentGreen) "red" else "green"
                            debugRatio = otherRatio // Show the ratio of the new color
                            // Log.d(TAG, "Swapped $clsName -> $newClsName (Ratio: $currentRatio vs $otherRatio)")
                        }
                    } else {
                        // Color validated
                    }
                }
            }

            val newBox = OverlayView.BoundingBox(box.box, newClsName, newScore)
            if (debugRatio != -1f) {
                newBox.debugRatio = debugRatio
            }
            correctedBoxes.add(newBox)
        }

        return correctedBoxes
    }

    private fun calculateColorRatio(pixels: IntArray, isGreen: Boolean): Float {
        var count = 0
        val hsv = FloatArray(3)

        for (color in pixels) {
            Color.colorToHSV(color, hsv)
            val h = hsv[0] // 0..360
            val s = hsv[1] // 0..1
            val v = hsv[2] // 0..1

            // S, V thresholds: Green: S>25, V>40 | Red: S>50, V>50

            if (isGreen) {
                if (h in 50f..200f && s >= 0.1f && v >= 0.15f) {
                    count++
                }
            } else { // Red
                if ((h <= 20f || h >= 340f) && s >= 0.2f && v >= 0.2f) {
                    count++
                }
            }
        }

        return if (pixels.isNotEmpty()) count.toFloat() / pixels.size else 0f
    }

    fun selectTargetTrafficLight(boxes: List<OverlayView.BoundingBox>): Pair<OverlayView.BoundingBox, Float>? {
        // Filter red/green only
        val trafficLights = boxes.filter {
            it.clsName.equals("red", ignoreCase = true) || it.clsName.equals(
                "green",
                ignoreCase = true
            )
        }

        if (trafficLights.isEmpty()) return null

        // Center is (0.5, 0.5) in normalized coordinates
        val centerX = 0.5f
        val centerY = 0.5f

        var bestBox: OverlayView.BoundingBox? = null
        var bestScore = -1f

        for (box in trafficLights) {
            val rect = box.box // Normalized RectF(0..1)

            val boxCx = rect.centerX()
            val boxCy = rect.centerY()

            // Euclidean distance to center
            val dist = Math.sqrt(
                Math.pow((boxCx - centerX).toDouble(), 2.0) +
                        Math.pow((boxCy - centerY).toDouble(), 2.0)
            ).toFloat()

            val area = rect.width() * rect.height()

            // Score Formula: Area / (Distance + epsilon)
            val score = area / (dist + 0.1f)

            if (score > bestScore) {
                bestScore = score
                bestBox = box
            }
        }

        if (bestBox != null) {
            Log.d(TAG, "Selected Target: ${bestBox.clsName} (Score: $bestScore)")
            return Pair(bestBox, bestScore)
        }

        return null
    }

    fun updateTrafficLightState(targetBox: OverlayView.BoundingBox?): TrafficLightState {
        // Determine raw input state
        val currentState = when {
            targetBox == null -> TrafficLightState.UNKNOWN
            targetBox.clsName.equals("red", ignoreCase = true) -> TrafficLightState.RED
            targetBox.clsName.equals("green", ignoreCase = true) -> TrafficLightState.GREEN
            else -> TrafficLightState.UNKNOWN
        }

        val currentTime = System.currentTimeMillis()

        // 1. Consecutive Detection Logic (Debouncing)
        if (currentState != TrafficLightState.UNKNOWN) {
            if (currentState == candidateState) {
                consecutiveCount++
            } else {
                candidateState = currentState
                consecutiveCount = 1
            }

            // If we have enough consistent frames, update the "Real" state
            if (consecutiveCount >= TRIGGER_THRESHOLD) {
                lastKnownState = currentState
                lastStateTimeTime = currentTime
            }
        }

        // 2. Persistence Logic (Handling Occlusion)
        // If current detection is lost (UNKNOWN), check if we can persist the old state
        return if (currentState == TrafficLightState.UNKNOWN) {
            if (currentTime - lastStateTimeTime < PERSISTENCE_DURATION_MS) {
                // Keep showing the last known state (e.g. Red) for a while
                lastKnownState
            } else {
                TrafficLightState.UNKNOWN
            }
        } else {
            // If currently detecting something, return the robust (debounced) state
            // Logic: We return lastKnownState which is only updated after N frames.
            // This prevents single-frame flickers.

            // Bug Fix: Only return lastKnownState if it is still valid (within persistence window).
            // If it's too old, we should show nothing (UNKNOWN) while verifying the new input.
            if (currentTime - lastStateTimeTime < PERSISTENCE_DURATION_MS) {
                lastKnownState
            } else {
                TrafficLightState.UNKNOWN
            }
        }
    }
}
