package kr.co.gachon.pproject6.via.ml

import android.graphics.RectF
import kr.co.gachon.pproject6.via.ui.OverlayView
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

class ObjectTracker {

    private var trackedTarget: OverlayView.BoundingBox? = null
    
    // IoU Threshold for tracking persistence
    private val IOU_THRESHOLD = 0.5f

    fun selectTarget(boxes: List<OverlayView.BoundingBox>): Pair<OverlayView.BoundingBox, Float>? {
        val trafficLights = boxes.filter { DetectionLabels.isPedestrianSignal(it.clsName) }

        if (trafficLights.isEmpty()) {
            trackedTarget = null
            return null
        }

        // 1. If we have a tracked target, try to find it in the new frame
        if (trackedTarget != null) {
            var bestIoU = -1f
            var bestMatch: OverlayView.BoundingBox? = null

            for (box in trafficLights) {
                val iou = calculateIoU(trackedTarget!!.box, box.box)
                if (iou > bestIoU) {
                    bestIoU = iou
                    bestMatch = box
                }
            }

            // If we found a good match (Same object moved slightly)
            if (bestMatch != null && bestIoU >= IOU_THRESHOLD) {
                // Update tracked target to the new box (position/score)
                // We recalculate score just for display, but logic prefers the tracked one
                val score = calculateScore(bestMatch)
                trackedTarget = bestMatch
                return Pair(bestMatch, score)
            }
        }

        // 2. If no tracking or lost tracking, find the best new target
        var bestBox: OverlayView.BoundingBox? = null
        var bestScore = -1f

        for (box in trafficLights) {
            val score = calculateScore(box)
            if (score > bestScore) {
                bestScore = score
                bestBox = box
            }
        }

        // Update tracking
        trackedTarget = bestBox
        
        return if (bestBox != null) {
            Pair(bestBox, bestScore)
        } else {
            null
        }
    }

    fun reset() {
        trackedTarget = null
    }

    private fun calculateScore(box: OverlayView.BoundingBox): Float {
        // Center is (0.5, 0.5) in normalized coordinates
        val centerX = 0.5f
        val centerY = 0.5f
        
        val rect = box.box
        val boxCx = rect.centerX()
        val boxCy = rect.centerY()

        // Euclidean distance to center
        val dist = Math.sqrt(
            Math.pow((boxCx - centerX).toDouble(), 2.0) +
                    Math.pow((boxCy - centerY).toDouble(), 2.0)
        ).toFloat()

        val area = rect.width() * rect.height()

        return calculateTargetPriority(
            confidence = box.score,
            normalizedArea = area,
            centerDistance = dist
        )
    }

    private fun calculateIoU(a: RectF, b: RectF): Float {
        val areaA = (a.right - a.left) * (a.bottom - a.top)
        val areaB = (b.right - b.left) * (b.bottom - b.top)

        val intersectionLeft = max(a.left, b.left)
        val intersectionTop = max(a.top, b.top)
        val intersectionRight = min(a.right, b.right)
        val intersectionBottom = min(a.bottom, b.bottom)

        if (intersectionLeft < intersectionRight && intersectionTop < intersectionBottom) {
            val intersectionArea =
                (intersectionRight - intersectionLeft) * (intersectionBottom - intersectionTop)
            return intersectionArea / (areaA + areaB - intersectionArea)
        }
        return 0f
    }
}

internal fun calculateTargetPriority(
    confidence: Float,
    normalizedArea: Float,
    centerDistance: Float
): Float {
    val clampedArea = normalizedArea.coerceAtLeast(1e-6f)
    val sizeWeightedArea = clampedArea.toDouble().pow(1.25).toFloat()
    // Center proximity still matters, but it now acts as a mild tie-breaker
    // instead of overwhelming size when a nearer crosswalk signal appears larger.
    val centerBias = 1f / (1f + centerDistance * 1.25f)
    return confidence * sizeWeightedArea * centerBias * 1000f
}
