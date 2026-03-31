package kr.co.gachon.pproject6.via.util

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect

object ImageUtils {
    private val bitmapPaint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)

    fun cropBitmap(bitmap: Bitmap, cropRect: Rect): Bitmap {
        val safeRect = Rect(
            cropRect.left.coerceIn(0, bitmap.width),
            cropRect.top.coerceIn(0, bitmap.height),
            cropRect.right.coerceIn(0, bitmap.width),
            cropRect.bottom.coerceIn(0, bitmap.height)
        )

        if (safeRect.width() <= 0 || safeRect.height() <= 0) {
            return bitmap
        }

        if (
            safeRect.left == 0 &&
            safeRect.top == 0 &&
            safeRect.right == bitmap.width &&
            safeRect.bottom == bitmap.height
        ) {
            return bitmap
        }

        return Bitmap.createBitmap(
            bitmap,
            safeRect.left,
            safeRect.top,
            safeRect.width(),
            safeRect.height()
        )
    }

    fun rotateBitmap(bitmap: Bitmap, degrees: Float, reusableBitmap: Bitmap? = null): Bitmap {
        val normalizedDegrees = ((degrees % 360f) + 360f) % 360f
        if (normalizedDegrees == 0f) return bitmap

        val swapDimensions = normalizedDegrees == 90f || normalizedDegrees == 270f
        val targetWidth = if (swapDimensions) bitmap.height else bitmap.width
        val targetHeight = if (swapDimensions) bitmap.width else bitmap.height
        val bitmapConfig = bitmap.config ?: Bitmap.Config.ARGB_8888

        val targetBitmap = if (
            reusableBitmap != null &&
            reusableBitmap.width == targetWidth &&
            reusableBitmap.height == targetHeight &&
            reusableBitmap.config == bitmapConfig &&
            reusableBitmap.isMutable
        ) {
            reusableBitmap
        } else {
            Bitmap.createBitmap(targetWidth, targetHeight, bitmapConfig)
        }

        targetBitmap.eraseColor(Color.TRANSPARENT)

        val matrix = Matrix().apply {
            postTranslate(-bitmap.width / 2f, -bitmap.height / 2f)
            postRotate(normalizedDegrees)
            postTranslate(targetWidth / 2f, targetHeight / 2f)
        }
        Canvas(targetBitmap).drawBitmap(bitmap, matrix, bitmapPaint)
        return targetBitmap
    }
}
