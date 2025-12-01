package kr.co.gachon.pproject6.via

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.PriorityQueue
import kotlin.math.max
import kotlin.math.min

class YoloDetector(
    private val context: Context,
    private val modelPath: String,
    private val useGpu: Boolean = false
) {

    private var interpreter: Interpreter? = null
    private var inputImageWidth = 0
    private var inputImageHeight = 0
    private var outputShape = intArrayOf()
    
    // Labels (COCO 80 classes)
    private val labels = listOf(
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    )

    fun setup() {
        val options = Interpreter.Options()
        if (useGpu) {
            if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                options.addDelegate(GpuDelegate())
            } else {
                // Fallback to CPU if GPU not supported
            }
        }
        options.setNumThreads(4)
        
        val model = FileUtil.loadMappedFile(context, modelPath)
        interpreter = Interpreter(model, options)

        val inputShape = interpreter!!.getInputTensor(0).shape() // [1, 640, 640, 3]
        inputImageWidth = inputShape[1]
        inputImageHeight = inputShape[2]
        
        val outputTensor = interpreter!!.getOutputTensor(0)
        outputShape = outputTensor.shape() // [1, 84, 8400] usually
    }

    fun detect(bitmap: Bitmap, confidenceThreshold: Float): DetectionResult {
        if (interpreter == null) return DetectionResult(emptyList(), 0, 0)

        val inferenceStartTime = SystemClock.uptimeMillis()

        // Preprocess
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(inputImageHeight, inputImageWidth, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f)) // Normalize to [0, 1]
            .build()

        var tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)
        tensorImage = imageProcessor.process(tensorImage)

        // Run inference
        val outputBuffer = TensorBuffer.createFixedSize(outputShape, DataType.FLOAT32)
        interpreter!!.run(tensorImage.buffer, outputBuffer.buffer.rewind())

        val inferenceTime = SystemClock.uptimeMillis() - inferenceStartTime

        // Post-process
        val outputArray = outputBuffer.floatArray
        val results = postProcess(outputArray, confidenceThreshold)

        return DetectionResult(results, inferenceTime)
    }

    private fun postProcess(output: FloatArray, threshold: Float): List<OverlayView.BoundingBox> {
        val boundingBoxes = mutableListOf<OverlayView.BoundingBox>()
        
        // Output shape is likely [1, 84, 8400]
        // 84 channels: 4 box coords + 80 classes
        // 8400 anchors
        
        val numChannels = outputShape[1] // 84
        val numAnchors = outputShape[2] // 8400
        
        // If shape is transposed [1, 8400, 84], adjust logic
        // But usually YOLOv8 export is [1, 84, 8400]
        
        val isTransposed = outputShape[1] > outputShape[2] // e.g. [1, 8400, 84]
        
        val rows = if (isTransposed) outputShape[1] else outputShape[2]
        val cols = if (isTransposed) outputShape[2] else outputShape[1]
        
        // Helper to access data
        fun get(row: Int, col: Int): Float {
            return if (isTransposed) {
                output[row * cols + col]
            } else {
                output[col * rows + row] // Transpose access: channel * numAnchors + anchorIndex
            }
        }

        for (i in 0 until rows) {
            // Find max score class
            var maxScore = 0f
            var maxClassIndex = -1
            
            // Classes start at index 4
            for (c in 0 until 80) {
                val score = get(i, 4 + c)
                if (score > maxScore) {
                    maxScore = score
                    maxClassIndex = c
                }
            }

            if (maxScore > threshold) {
                var cx = get(i, 0)
                var cy = get(i, 1)
                var w = get(i, 2)
                var h = get(i, 3)

                // Debug log for the first detection to check values
                if (boundingBoxes.isEmpty()) {
                    android.util.Log.d("YoloDetector", "Raw coords: cx=$cx, cy=$cy, w=$w, h=$h, inputSize=$inputImageWidth x $inputImageHeight")
                }

                // Heuristic: If coordinates are in pixels (likely > 1.0), normalize them.
                // If they are already normalized (0..1), keep them as is.
                // We assume that if any value is significantly larger than 1.0, it's in pixels.
                if (cx > 1.0f || cy > 1.0f || w > 1.0f || h > 1.0f) {
                    cx /= inputImageWidth
                    cy /= inputImageHeight
                    w /= inputImageWidth
                    h /= inputImageHeight
                }

                val left = cx - w / 2
                val top = cy - h / 2
                val right = cx + w / 2
                val bottom = cy + h / 2

                val rect = RectF(
                    max(0f, left),
                    max(0f, top),
                    min(1f, right),
                    min(1f, bottom)
                )
                
                boundingBoxes.add(OverlayView.BoundingBox(rect, labels.getOrElse(maxClassIndex) { "Unknown" }, maxScore))
            }
        }

        return nms(boundingBoxes)
    }

    private fun nms(boxes: List<OverlayView.BoundingBox>): List<OverlayView.BoundingBox> {
        val pq = PriorityQueue<OverlayView.BoundingBox> { o1, o2 -> o2.score.compareTo(o1.score) }
        pq.addAll(boxes)

        val selected = mutableListOf<OverlayView.BoundingBox>()
        
        while (pq.isNotEmpty()) {
            val best = pq.poll()
            selected.add(best)

            val iterator = pq.iterator()
            while (iterator.hasNext()) {
                val other = iterator.next()
                if (iou(best.box, other.box) > 0.45f) { // IOU Threshold
                    iterator.remove()
                }
            }
        }
        return selected
    }

    private fun iou(a: RectF, b: RectF): Float {
        val areaA = (a.right - a.left) * (a.bottom - a.top)
        val areaB = (b.right - b.left) * (b.bottom - b.top)

        val intersectionLeft = max(a.left, b.left)
        val intersectionTop = max(a.top, b.top)
        val intersectionRight = min(a.right, b.right)
        val intersectionBottom = min(a.bottom, b.bottom)

        if (intersectionLeft < intersectionRight && intersectionTop < intersectionBottom) {
            val intersectionArea = (intersectionRight - intersectionLeft) * (intersectionBottom - intersectionTop)
            return intersectionArea / (areaA + areaB - intersectionArea)
        }
        return 0f
    }

    fun close() {
        interpreter?.close()
    }

    data class DetectionResult(
        val boxes: List<OverlayView.BoundingBox>,
        val inferenceTime: Long,
        val hardwareType: Int // 0: CPU, 1: GPU
    )
}
