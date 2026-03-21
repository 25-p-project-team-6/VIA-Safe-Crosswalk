package kr.co.gachon.pproject6.via.ml

import kr.co.gachon.pproject6.via.ui.OverlayView

class RiskObjectEvaluator(
    private val config: RiskObjectConfig = RiskObjectConfig()
) {
    fun findBlockingRisks(boxes: List<OverlayView.BoundingBox>): List<OverlayView.BoundingBox> {
        return boxes.filter { box ->
            val centerX = box.box.centerX()
            val area = box.box.width() * box.box.height()

            isBlockingRisk(
                label = box.clsName,
                score = box.score,
                centerX = centerX,
                bottom = box.box.bottom,
                area = area
            )
        }
    }

    fun isBlockingRisk(
        label: String,
        score: Float,
        centerX: Float,
        bottom: Float,
        area: Float
    ): Boolean {
        return label.lowercase() in config.blockingLabels &&
            score >= config.minScore &&
            centerX in config.blockingCenterBand &&
            (bottom >= config.minBottom || area >= config.minArea)
    }
}

data class RiskObjectConfig(
    val blockingLabels: Set<String> = setOf("bicycle", "car", "motorcycle", "bus", "train", "truck"),
    val minScore: Float = 0.35f,
    val minBottom: Float = 0.45f,
    val minArea: Float = 0.015f,
    val blockingCenterBand: ClosedFloatingPointRange<Float> = 0.2f..0.8f
)
