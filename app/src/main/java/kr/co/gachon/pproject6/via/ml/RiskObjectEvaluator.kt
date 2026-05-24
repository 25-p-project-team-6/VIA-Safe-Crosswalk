package kr.co.gachon.pproject6.via.ml

import kr.co.gachon.pproject6.via.ui.OverlayView

class CrosswalkOccupancyEvaluator(
    private val config: CrosswalkOccupancyConfig = CrosswalkOccupancyConfig(),
    private val timeProvider: () -> Long = System::currentTimeMillis
) {
    private var candidateStartedAt: Long = Long.MIN_VALUE

    fun findActiveOccupancy(
        boxes: List<OverlayView.BoundingBox>,
        eligible: Boolean
    ): List<OverlayView.BoundingBox> {
        if (!eligible) {
            reset()
            return emptyList()
        }

        val candidates =
            boxes.filter { box ->
                val centerX = box.box.centerX()
                val area = box.box.width() * box.box.height()
                isOccupancyCandidate(
                    label = box.clsName,
                    score = box.score,
                    centerX = centerX,
                    bottom = box.box.bottom,
                    area = area
                )
            }

        if (candidates.isEmpty()) {
            reset()
            return emptyList()
        }

        val now = timeProvider()
        if (candidateStartedAt == Long.MIN_VALUE) {
            candidateStartedAt = now
            return emptyList()
        }

        return if (now - candidateStartedAt >= config.confirmDurationMs) {
            candidates
        } else {
            emptyList()
        }
    }

    fun isOccupancyCandidate(
        label: String,
        score: Float,
        centerX: Float,
        bottom: Float,
        area: Float
    ): Boolean {
        return label.lowercase() in config.labels &&
            score >= config.minScore &&
            centerX in config.centerBand &&
            (bottom >= config.minBottom || area >= config.minArea)
    }

    fun reset() {
        candidateStartedAt = Long.MIN_VALUE
    }
}

data class CrosswalkOccupancyConfig(
    val labels: Set<String> = DetectionLabels.occupancyLabels,
    val minScore: Float = 0.35f,
    val minBottom: Float = 0.45f,
    val minArea: Float = 0.015f,
    val centerBand: ClosedFloatingPointRange<Float> = 0.2f..0.8f,
    val confirmDurationMs: Long = 400L
)
