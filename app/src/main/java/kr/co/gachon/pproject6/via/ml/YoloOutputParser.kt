package kr.co.gachon.pproject6.via.ml

import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

data class ParsedYoloDetection(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val clsName: String,
    val score: Float
)

object YoloOutputParser {
    fun layoutName(outputCols: Int, labels: List<String>): String {
        return if (isBatchedNmsLayout(outputCols, labels)) {
            "batched_nms_xyxy_score_class"
        } else {
            "class_score_cxcywh"
        }
    }

    fun parse(
        output: FloatArray,
        outputRows: Int,
        outputCols: Int,
        outputIsTransposed: Boolean,
        inputImageWidth: Int,
        inputImageHeight: Int,
        labels: List<String>,
        threshold: Float,
        specificConfidenceThresholds: Map<String, Float> = emptyMap()
    ): List<ParsedYoloDetection> {
        val detections = mutableListOf<ParsedYoloDetection>()

        fun get(row: Int, col: Int): Float {
            return if (outputIsTransposed) {
                output[row * outputCols + col]
            } else {
                output[col * outputRows + row]
            }
        }

        for (row in 0 until outputRows) {
            val detection =
                if (isBatchedNmsLayout(outputCols, labels)) {
                    parseBatchedNmsRow(
                        row = row,
                        get = ::get,
                        inputImageWidth = inputImageWidth,
                        inputImageHeight = inputImageHeight,
                        labels = labels,
                        threshold = threshold,
                        specificConfidenceThresholds = specificConfidenceThresholds
                    )
                } else {
                    parseClassScoreRow(
                        row = row,
                        outputCols = outputCols,
                        get = ::get,
                        inputImageWidth = inputImageWidth,
                        inputImageHeight = inputImageHeight,
                        labels = labels,
                        threshold = threshold,
                        specificConfidenceThresholds = specificConfidenceThresholds
                    )
                }
            detection?.let(detections::add)
        }

        return detections
    }

    private fun isBatchedNmsLayout(outputCols: Int, labels: List<String>): Boolean {
        // Ultralytics TFLite NMS exports commonly emit [x1, y1, x2, y2, score, class].
        // Legacy/class-score outputs with two labels can also have 6 columns
        // ([cx, cy, w, h, class0, class1]), so require more labels than score columns.
        return outputCols == 6 && labels.size > outputCols - 4
    }

    private fun parseBatchedNmsRow(
        row: Int,
        get: (Int, Int) -> Float,
        inputImageWidth: Int,
        inputImageHeight: Int,
        labels: List<String>,
        threshold: Float,
        specificConfidenceThresholds: Map<String, Float>
    ): ParsedYoloDetection? {
        val score = get(row, 4)
        val classIndex = get(row, 5).roundToInt()
        if (classIndex !in labels.indices || !score.isFinite()) {
            return null
        }

        val clsName = labels[classIndex]
        val confThreshold = specificConfidenceThresholds[clsName] ?: threshold
        if (score <= confThreshold) {
            return null
        }

        return normalizedXyxyDetection(
            first = get(row, 0),
            second = get(row, 1),
            third = get(row, 2),
            fourth = get(row, 3),
            inputImageWidth = inputImageWidth,
            inputImageHeight = inputImageHeight,
            clsName = clsName,
            score = score
        )
    }

    private fun parseClassScoreRow(
        row: Int,
        outputCols: Int,
        get: (Int, Int) -> Float,
        inputImageWidth: Int,
        inputImageHeight: Int,
        labels: List<String>,
        threshold: Float,
        specificConfidenceThresholds: Map<String, Float>
    ): ParsedYoloDetection? {
        var maxScore = 0f
        var maxClassIndex = -1
        val numClasses = outputCols - 4
        for (classOffset in 0 until numClasses) {
            val score = get(row, 4 + classOffset)
            if (score > maxScore) {
                maxScore = score
                maxClassIndex = classOffset
            }
        }

        if (maxClassIndex !in labels.indices || maxScore <= 0.1f || !maxScore.isFinite()) {
            return null
        }

        val clsName = labels[maxClassIndex]
        val confThreshold = specificConfidenceThresholds[clsName] ?: threshold
        if (maxScore <= confThreshold) {
            return null
        }

        return normalizedCxcywhDetection(
            cx = get(row, 0),
            cy = get(row, 1),
            width = get(row, 2),
            height = get(row, 3),
            inputImageWidth = inputImageWidth,
            inputImageHeight = inputImageHeight,
            clsName = clsName,
            score = maxScore
        )
    }

    private fun normalizedXyxyDetection(
        first: Float,
        second: Float,
        third: Float,
        fourth: Float,
        inputImageWidth: Int,
        inputImageHeight: Int,
        clsName: String,
        score: Float
    ): ParsedYoloDetection? {
        var left = first
        var top = second
        var right = third
        var bottom = fourth

        if (left > 1f || top > 1f || right > 1f || bottom > 1f) {
            left /= inputImageWidth
            right /= inputImageWidth
            top /= inputImageHeight
            bottom /= inputImageHeight
        }

        if (right <= left || bottom <= top) {
            return normalizedCxcywhDetection(
                cx = first,
                cy = second,
                width = third,
                height = fourth,
                inputImageWidth = inputImageWidth,
                inputImageHeight = inputImageHeight,
                clsName = clsName,
                score = score
            )
        }

        return normalizedDetection(left, top, right, bottom, clsName, score)
    }

    private fun normalizedCxcywhDetection(
        cx: Float,
        cy: Float,
        width: Float,
        height: Float,
        inputImageWidth: Int,
        inputImageHeight: Int,
        clsName: String,
        score: Float
    ): ParsedYoloDetection? {
        var normalizedCx = cx
        var normalizedCy = cy
        var normalizedWidth = width
        var normalizedHeight = height

        if (normalizedCx > 1f || normalizedCy > 1f || normalizedWidth > 1f || normalizedHeight > 1f) {
            normalizedCx /= inputImageWidth
            normalizedCy /= inputImageHeight
            normalizedWidth /= inputImageWidth
            normalizedHeight /= inputImageHeight
        }

        val left = normalizedCx - normalizedWidth / 2
        val top = normalizedCy - normalizedHeight / 2
        val right = normalizedCx + normalizedWidth / 2
        val bottom = normalizedCy + normalizedHeight / 2
        return normalizedDetection(left, top, right, bottom, clsName, score)
    }

    private fun normalizedDetection(
        left: Float,
        top: Float,
        right: Float,
        bottom: Float,
        clsName: String,
        score: Float
    ): ParsedYoloDetection? {
        val clampedLeft = max(0f, left)
        val clampedTop = max(0f, top)
        val clampedRight = min(1f, right)
        val clampedBottom = min(1f, bottom)
        if (clampedRight <= clampedLeft || clampedBottom <= clampedTop) {
            return null
        }
        return ParsedYoloDetection(
            left = clampedLeft,
            top = clampedTop,
            right = clampedRight,
            bottom = clampedBottom,
            clsName = clsName,
            score = score
        )
    }
}
