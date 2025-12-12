package kr.co.gachon.pproject6.via

import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log

object PostProcessor {
    private const val TAG = "PostProcessor"

    fun applyColorCorrection(
        bitmap: Bitmap,
        boxes: List<OverlayView.BoundingBox>
    ): List<OverlayView.BoundingBox> {
        val correctedBoxes = mutableListOf<OverlayView.BoundingBox>()

        for (box in boxes) {
            var clsName = box.clsName
            var newClsName = clsName
            val newScore = box.score // Keep confidence score

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

                    if (currentRatio <= 0.05f) {
                        // Check other color
                        val otherRatio = calculateColorRatio(pixels, !isCurrentGreen)

                        if (otherRatio > 0.05f) {
                            // Swap!
                            newClsName = if (isCurrentGreen) "red" else "green"
                            // Log.d(TAG, "Swapped $clsName -> $newClsName (Ratio: $currentRatio vs $otherRatio)")
                        }
                    } else {
                        // Log.d(TAG, "Validated $clsName (Ratio: $currentRatio)")
                    }
                }
            }

            correctedBoxes.add(OverlayView.BoundingBox(box.box, newClsName, newScore))
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

            // Notebook Green: H[25..100] (OpenCV 0-180) -> H[50..200] (Android 0-360)
            // Notebook Red: H[0..10] U [170..180] -> H[0..20] U [340..360]

            // S, V thresholds:
            // Green: S>25 (~0.1), V>40 (~0.15)
            // Red: S>50 (~0.2), V>50 (~0.2)

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

    fun selectTargetTrafficLight(boxes: List<OverlayView.BoundingBox>): OverlayView.BoundingBox? {
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
            // Larger Area -> Higher Score
            // Smaller Distance -> Higher Score
            // Epsilon prevents division by zero (though dist 0 is unlikely)
            val score = area / (dist + 0.1f)

            if (score > bestScore) {
                bestScore = score
                bestBox = box
            }

            // Log details for debugging logic
            // Log.d(TAG, "Candidate: ${box.clsName}, Area: $area, Dist: $dist, Score: $score")
        }

        if (bestBox != null) {
            Log.d(TAG, "Selected Target: ${bestBox.clsName} (Score: $bestScore)")
        }

        return bestBox
    }
}
