package kr.co.gachon.pproject6.via

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import java.util.LinkedList
import kotlin.math.max
import kotlin.math.min

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results: List<BoundingBox> = LinkedList()
    private var boxPaint = Paint()
    private var textBackgroundPaint = Paint()
    private var textPaint = Paint()
    
    private var sourceWidth: Int = 0
    private var sourceHeight: Int = 0

    init {
        initPaints()
    }

    fun clear() {
        textPaint.reset()
        textBackgroundPaint.reset()
        boxPaint.reset()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        textBackgroundPaint.color = Color.BLACK
        textBackgroundPaint.style = Paint.Style.FILL
        textBackgroundPaint.textSize = 50f

        textPaint.color = Color.WHITE
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 50f

        boxPaint.color = ContextCompat.getColor(context!!, android.R.color.holo_red_dark)
        boxPaint.strokeWidth = 8F
        boxPaint.style = Paint.Style.STROKE
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (sourceWidth == 0 || sourceHeight == 0) return

        // Calculate scale and offset to match PreviewView's FILL_CENTER
        // PreviewView scales the image to fill the view, cropping if necessary.
        val viewWidth = width.toFloat()
        val viewHeight = height.toFloat()
        
        val scale = max(viewWidth / sourceWidth, viewHeight / sourceHeight)
        val dx = (viewWidth - sourceWidth * scale) / 2
        val dy = (viewHeight - sourceHeight * scale) / 2

        for (result in results) {
            val boundingBox = result.box
            
            // Transform normalized coordinates to view coordinates
            // 1. Denormalize to source image coordinates
            val topPx = boundingBox.top * sourceHeight
            val bottomPx = boundingBox.bottom * sourceHeight
            val leftPx = boundingBox.left * sourceWidth
            val rightPx = boundingBox.right * sourceWidth
            
            // 2. Apply scale and offset
            val top = topPx * scale + dy
            val bottom = bottomPx * scale + dy
            val left = leftPx * scale + dx
            val right = rightPx * scale + dx

            // Draw bounding box
            val drawableRect = RectF(left, top, right, bottom)
            canvas.drawRect(drawableRect, boxPaint)

            // Create text to display
            val drawableText = "${result.clsName} ${String.format("%.2f", result.score)}"

            // Draw rect behind display text
            val textWidth = textPaint.measureText(drawableText)
            val textHeight = textPaint.textSize
            canvas.drawRect(
                left,
                top,
                left + textWidth + 8,
                top + textHeight + 8,
                textBackgroundPaint
            )

            // Draw text
            canvas.drawText(drawableText, left, top + textHeight, textPaint)
        }
    }

    fun setResults(boundingBoxes: List<BoundingBox>) {
        results = boundingBoxes
        invalidate()
    }
    
    fun setInputImageSize(width: Int, height: Int) {
        this.sourceWidth = width
        this.sourceHeight = height
    }

    data class BoundingBox(
        val box: RectF,
        val clsName: String,
        val score: Float
    )
}
