package kr.co.gachon.pproject6.via.ml

import android.graphics.Rect
import android.graphics.RectF
import kotlin.math.min

/**
 * Keeps YOLO's square tensor input without shrinking the central scene.
 *
 * The camera/replay frame can be rectangular, but the model expects a square image.
 * Letterbox preserves the full frame but makes small signal lights smaller. For this
 * demo path, we center-crop the longer axis and resize the square crop to the model
 * input. Detector outputs are then mapped back to the original source bitmap space.
 */
data class SquareCropTransform(
    val sourceWidth: Int,
    val sourceHeight: Int,
    val inputWidth: Int,
    val inputHeight: Int,
    val cropLeft: Int,
    val cropTop: Int,
    val cropSize: Int
) {
    val sourceRect: Rect = Rect(cropLeft, cropTop, cropLeft + cropSize, cropTop + cropSize)
    val destinationRect: RectF = RectF(0f, 0f, inputWidth.toFloat(), inputHeight.toFloat())

    fun inputRectToSourceNormalized(inputNormalizedRect: RectF): RectF? {
        return inputBoundsToSourceNormalized(
            left = inputNormalizedRect.left,
            top = inputNormalizedRect.top,
            right = inputNormalizedRect.right,
            bottom = inputNormalizedRect.bottom
        )?.toRectF()
    }

    internal fun inputBoundsToSourceNormalized(
        left: Float,
        top: Float,
        right: Float,
        bottom: Float
    ): NormalizedBox? {
        if (sourceWidth <= 0 || sourceHeight <= 0 || inputWidth <= 0 || inputHeight <= 0 || cropSize <= 0) {
            return null
        }

        val sourceLeft = cropLeft + (left * cropSize)
        val sourceTop = cropTop + (top * cropSize)
        val sourceRight = cropLeft + (right * cropSize)
        val sourceBottom = cropTop + (bottom * cropSize)

        val mappedLeft = (sourceLeft / sourceWidth).coerceIn(0f, 1f)
        val mappedTop = (sourceTop / sourceHeight).coerceIn(0f, 1f)
        val mappedRight = (sourceRight / sourceWidth).coerceIn(0f, 1f)
        val mappedBottom = (sourceBottom / sourceHeight).coerceIn(0f, 1f)

        if (mappedRight <= mappedLeft || mappedBottom <= mappedTop) {
            return null
        }

        return NormalizedBox(mappedLeft, mappedTop, mappedRight, mappedBottom)
    }

    companion object {
        fun from(
            sourceWidth: Int,
            sourceHeight: Int,
            inputWidth: Int,
            inputHeight: Int
        ): SquareCropTransform {
            require(sourceWidth > 0) { "sourceWidth must be positive" }
            require(sourceHeight > 0) { "sourceHeight must be positive" }
            require(inputWidth > 0) { "inputWidth must be positive" }
            require(inputHeight > 0) { "inputHeight must be positive" }

            val cropSize = min(sourceWidth, sourceHeight)
            val cropLeft = (sourceWidth - cropSize) / 2
            val cropTop = (sourceHeight - cropSize) / 2

            return SquareCropTransform(
                sourceWidth = sourceWidth,
                sourceHeight = sourceHeight,
                inputWidth = inputWidth,
                inputHeight = inputHeight,
                cropLeft = cropLeft,
                cropTop = cropTop,
                cropSize = cropSize
            )
        }
    }
}

data class NormalizedBox(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float
) {
    fun toRectF(): RectF = RectF(left, top, right, bottom)
}
