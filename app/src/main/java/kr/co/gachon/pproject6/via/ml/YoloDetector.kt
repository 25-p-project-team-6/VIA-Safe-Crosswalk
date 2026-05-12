package kr.co.gachon.pproject6.via.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Rect
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import kr.co.gachon.pproject6.via.ui.OverlayView
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.PriorityQueue
import kotlin.math.roundToInt
import kotlin.math.max
import kotlin.math.min

// Labels (COCO 80 classes) - Fallback if no labels provided
private val cocoLabels = listOf(
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush"
)

class YoloDetector(
    private val context: Context,
    private val modelPath: String,
    private val useGpu: Boolean = false,
    private val labels: List<String> = cocoLabels,
    private val defaultIouThreshold: Float = 0.5f,
    private val specificIouThresholds: Map<String, Float> = emptyMap()
) {
    companion object {
        private const val TAG = "VIA_GPU"
    }

    var specificConfidenceThresholds: Map<String, Float> = emptyMap()

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var inputImageWidth = 0
    private var inputImageHeight = 0
    private var outputShape = intArrayOf()
    private var outputRows = 0
    private var outputCols = 0
    private var outputIsTransposed = false
    private var inputDataType: DataType = DataType.FLOAT32
    private var outputDataType: DataType = DataType.FLOAT32
    private var inputQuantization = Tensor.QuantizationParams(0f, 0)
    private var outputQuantization = Tensor.QuantizationParams(0f, 0)
    private var inputBuffer: ByteBuffer? = null
    private var outputBuffer: ByteBuffer? = null
    private var scaledBitmapBuffer: Bitmap? = null
    private var pixelBuffer: IntArray = intArrayOf()
    var runtimeBackendLabel: String = "CPU"
        private set
    var compatibilityReportedSupported: Boolean = false
        private set

    fun setup() {
        val options = Interpreter.Options()
        compatibilityReportedSupported = CompatibilityList().isDelegateSupportedOnThisDevice
        if (useGpu) {
            try {
                gpuDelegate = GpuDelegate().also {
                    runtimeBackendLabel = "GPU"
                }
                options.addDelegate(gpuDelegate)
            } catch (gpuError: Exception) {
                runtimeBackendLabel = "CPU"
                gpuDelegate?.close()
                gpuDelegate = null
                Log.i(
                    TAG,
                    "gpu_fallback model=$modelPath compat=$compatibilityReportedSupported"
                )
            }
        } else {
            runtimeBackendLabel = "CPU"
        }
        options.setNumThreads(4)

        val model = loadMappedAsset(modelPath)
        interpreter = Interpreter(model, options)

        val inputTensor = interpreter!!.getInputTensor(0)
        val inputShape = inputTensor.shape() // [1, 640, 640, 3]
        inputImageWidth = inputShape[1]
        inputImageHeight = inputShape[2]
        inputDataType = inputTensor.dataType()
        inputQuantization = inputTensor.quantizationParams()

        val outputTensor = interpreter!!.getOutputTensor(0)
        outputShape = outputTensor.shape() // [1, 84, 8400] usually
        outputDataType = outputTensor.dataType()
        outputQuantization = outputTensor.quantizationParams()
        outputIsTransposed = outputShape[1] > outputShape[2]
        outputRows = if (outputIsTransposed) outputShape[1] else outputShape[2]
        outputCols = if (outputIsTransposed) outputShape[2] else outputShape[1]
        inputBuffer =
            ByteBuffer.allocateDirect(inputShape.product() * inputDataType.byteSize())
                .order(ByteOrder.nativeOrder())
        outputBuffer =
            ByteBuffer.allocateDirect(outputShape.product() * outputDataType.byteSize())
                .order(ByteOrder.nativeOrder())
        pixelBuffer = IntArray(inputImageWidth * inputImageHeight)
    }

    fun detect(bitmap: Bitmap, confidenceThreshold: Float): DetectionResult {
        val activeInterpreter = interpreter ?: return DetectionResult(emptyList(), 0)
        val activeInputBuffer = inputBuffer ?: return DetectionResult(emptyList(), 0)
        val activeOutputBuffer = outputBuffer ?: return DetectionResult(emptyList(), 0)

        val inferenceStartTime = SystemClock.uptimeMillis()

        fillInputBuffer(bitmap, activeInputBuffer)
        activeOutputBuffer.rewind()

        // Run inference
        activeInterpreter.run(activeInputBuffer, activeOutputBuffer)

        val inferenceTime = SystemClock.uptimeMillis() - inferenceStartTime

        // Post-process
        val outputArray = outputBufferToFloatArray(activeOutputBuffer)
        val results = postProcess(outputArray, confidenceThreshold)

        // Apply strict NMS first to reduce boxes
        val nmsResults = nms(results)

        // Return raw NMS results (no color correction inside detector)
        return DetectionResult(nmsResults, inferenceTime)
    }

    private fun loadMappedAsset(assetPath: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(assetPath)
        return FileInputStream(fileDescriptor.fileDescriptor).use { inputStream ->
            inputStream.channel.map(
                FileChannel.MapMode.READ_ONLY,
                fileDescriptor.startOffset,
                fileDescriptor.declaredLength
            )
        }
    }

    private fun fillInputBuffer(bitmap: Bitmap, buffer: ByteBuffer) {
        buffer.rewind()
        val scaledBitmap =
            if (bitmap.width == inputImageWidth && bitmap.height == inputImageHeight) {
                bitmap
            } else {
                val reusable =
                    scaledBitmapBuffer?.takeIf {
                        !it.isRecycled &&
                            it.width == inputImageWidth &&
                            it.height == inputImageHeight
                    } ?: Bitmap.createBitmap(
                        inputImageWidth,
                        inputImageHeight,
                        Bitmap.Config.ARGB_8888
                    ).also { scaledBitmapBuffer = it }
                Canvas(reusable).drawBitmap(
                    bitmap,
                    null,
                    Rect(0, 0, inputImageWidth, inputImageHeight),
                    null
                )
                reusable
            }
        scaledBitmapBuffer?.let { previous ->
            if (scaledBitmap !== previous &&
                previous.width == inputImageWidth &&
                previous.height == inputImageHeight
            ) {
                previous.recycle()
                scaledBitmapBuffer = null
            }
        }
        if (scaledBitmap !== bitmap) {
            scaledBitmapBuffer = scaledBitmap
        }
        scaledBitmap.getPixels(
            pixelBuffer,
            0,
            inputImageWidth,
            0,
            0,
            inputImageWidth,
            inputImageHeight
        )
        pixelBuffer.forEach { pixel ->
            val red = (pixel shr 16) and 0xFF
            val green = (pixel shr 8) and 0xFF
            val blue = pixel and 0xFF
            putInputChannel(buffer, red)
            putInputChannel(buffer, green)
            putInputChannel(buffer, blue)
        }
        buffer.rewind()
    }

    private fun putInputChannel(buffer: ByteBuffer, value: Int) {
        when (inputDataType) {
            DataType.FLOAT32 -> buffer.putFloat(value / 255f)
            DataType.UINT8 -> buffer.put(value.toByte())
            DataType.INT8 -> {
                val scale = inputQuantization.scale.takeIf { it > 0f } ?: (1f / 255f)
                val quantized = ((value / 255f) / scale + inputQuantization.zeroPoint)
                    .roundToInt()
                    .coerceIn(Byte.MIN_VALUE.toInt(), Byte.MAX_VALUE.toInt())
                buffer.put(quantized.toByte())
            }
            else -> error("Unsupported input type: $inputDataType")
        }
    }

    private fun outputBufferToFloatArray(buffer: ByteBuffer): FloatArray {
        buffer.rewind()
        val output = FloatArray(outputShape.product())
        when (outputDataType) {
            DataType.FLOAT32 -> {
                val floatBuffer = buffer.asFloatBuffer()
                floatBuffer.get(output)
            }
            DataType.UINT8 -> {
                val scale = outputQuantization.scale.takeIf { it > 0f } ?: 1f
                repeat(output.size) { index ->
                    val value = buffer.get().toInt() and 0xFF
                    output[index] = (value - outputQuantization.zeroPoint) * scale
                }
            }
            DataType.INT8 -> {
                val scale = outputQuantization.scale.takeIf { it > 0f } ?: 1f
                repeat(output.size) { index ->
                    val value = buffer.get().toInt()
                    output[index] = (value - outputQuantization.zeroPoint) * scale
                }
            }
            else -> error("Unsupported output type: $outputDataType")
        }
        buffer.rewind()
        return output
    }

    private fun postProcess(output: FloatArray, threshold: Float): List<OverlayView.BoundingBox> {
        val boundingBoxes = mutableListOf<OverlayView.BoundingBox>()

        // Helper to access data handling both NCHW and NHWC formats
        fun get(row: Int, col: Int): Float {
            return if (outputIsTransposed) {
                output[row * outputCols + col]
            } else {
                output[col * outputRows + row]
            }
        }

        for (i in 0 until outputRows) {
            // Find max score class
            var maxScore = 0f
            var maxClassIndex = -1

            // Classes start at index 4
            val numClasses = outputCols - 4
            for (c in 0 until numClasses) {
                val score = get(i, 4 + c)
                if (score > maxScore) {
                    maxScore = score
                    maxClassIndex = c
                }
            }

            // Optimization: Skip obvious background (score < 0.1)
            if (maxScore > 0.1f) {
                val clsName = labels.getOrElse(maxClassIndex) { "Unknown" }
                val confThreshold = specificConfidenceThresholds[clsName] ?: threshold

                if (maxScore > confThreshold) {
                    var cx = get(i, 0)
                    var cy = get(i, 1)
                    var w = get(i, 2)
                    var h = get(i, 3)

                    // Normalize coordinates if they are in pixels
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

                    boundingBoxes.add(OverlayView.BoundingBox(rect, clsName, maxScore))
                }
            }
        }

        return boundingBoxes // nms is called in detect now
    }

    private fun nms(boxes: List<OverlayView.BoundingBox>): List<OverlayView.BoundingBox> {
        val pq = PriorityQueue<OverlayView.BoundingBox> { o1, o2 -> o2.score.compareTo(o1.score) }
        pq.addAll(boxes)

        val selected = mutableListOf<OverlayView.BoundingBox>()

        while (pq.isNotEmpty()) {
            val best = pq.poll()
            selected.add(best!!)

            // Determine IOU threshold for this class
            val threshold = specificIouThresholds[best.clsName] ?: defaultIouThreshold

            val iterator = pq.iterator()
            while (iterator.hasNext()) {
                val other = iterator.next()
                if (iou(best.box, other.box) > threshold) {
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
            val intersectionArea =
                (intersectionRight - intersectionLeft) * (intersectionBottom - intersectionTop)
            return intersectionArea / (areaA + areaB - intersectionArea)
        }
        return 0f
    }

    fun close() {
        interpreter?.close()
        interpreter = null
        gpuDelegate?.close()
        gpuDelegate = null
        scaledBitmapBuffer?.recycle()
        scaledBitmapBuffer = null
        inputBuffer = null
        outputBuffer = null
    }

    data class DetectionResult(
        val boxes: List<OverlayView.BoundingBox>,
        val inferenceTime: Long
    )
}

private fun IntArray.product(): Int = fold(1) { acc, value -> acc * value }

private fun DataType.byteSize(): Int {
    return when (this) {
        DataType.FLOAT32 -> 4
        DataType.INT32 -> 4
        DataType.UINT8 -> 1
        DataType.INT8 -> 1
        DataType.INT64 -> 8
        DataType.BOOL -> 1
        else -> error("Unsupported tensor type: $this")
    }
}
