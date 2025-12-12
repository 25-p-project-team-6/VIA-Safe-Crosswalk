package kr.co.gachon.pproject6.via

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.android.material.slider.Slider
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import org.tensorflow.lite.gpu.CompatibilityList

import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat

class MainActivity : AppCompatActivity() {
    private lateinit var viewFinder: PreviewView
    private lateinit var overlay: OverlayView
    private lateinit var fpsText: TextView
    private lateinit var avgFpsText: TextView
    private lateinit var latencyText: TextView
    private lateinit var avgLatencyText: TextView
    private lateinit var modelNameText: TextView
    private lateinit var confidenceSlider: Slider
    private lateinit var gpuSwitch: com.google.android.material.switchmaterial.SwitchMaterial
    private lateinit var debugContainer: android.widget.LinearLayout
    private lateinit var debugToggleButton: android.widget.ImageButton

    // Set this to false to hide debug info (FPS, Latency, Hardware, Slider)
    private var showDebugInfo = true

    // Set this to false to hide bounding boxes and labels
    private val showBBoxOverlay = true

    private var cameraExecutor: ExecutorService? = null

    @Volatile
    private var detector: YoloDetector? = null
    private var confidenceThreshold = 0.5f

    private var lastFpsTimestamp = System.currentTimeMillis()
    private var frameCount = 0

    // Store timestamps and latencies for 10s sliding window
    // Pair(timestamp, latency)
    private val frameData = java.util.ArrayDeque<Pair<Long, Long>>()

    // We don't need totalFrameCount and startTime for the simple average anymore, 
    // but useful if we want total session average. 
    // However, user requested 10s average.
    // private var totalFrameCount = 0L // Removed/Unused for 10s avg
    // private var startTime = 0L // Removed/Unused for 10s avg

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
            if (isGranted) {
                startCamera()
            } else {
                Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
                finish()
            }
        }

    // traffic lights fine-tuned model label
    private val finetunedLabels =
        listOf("bicycle", "car", "motorcycle", "bus", "truck", "red", "green")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Handle Window Insets for edge-to-edge
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        viewFinder = findViewById(R.id.viewFinder)
        overlay = findViewById(R.id.overlay)
        overlay = findViewById(R.id.overlay)
        debugContainer = findViewById(R.id.debugContainer)
        debugToggleButton = findViewById(R.id.debugToggleButton)
        modelNameText = findViewById(R.id.modelNameText)
        fpsText = findViewById(R.id.fpsText)
        avgFpsText = findViewById(R.id.avgFpsText)
        latencyText = findViewById(R.id.latencyText)
        avgLatencyText = findViewById(R.id.avgLatencyText)
        confidenceSlider = findViewById(R.id.confidenceSlider)
        gpuSwitch = findViewById(R.id.gpuSwitch)

        // startTime = System.currentTimeMillis()

        debugContainer.visibility =
            if (showDebugInfo) android.view.View.VISIBLE else android.view.View.GONE

        debugToggleButton.setOnClickListener {
            showDebugInfo = !showDebugInfo
            debugContainer.visibility =
                if (showDebugInfo) android.view.View.VISIBLE else android.view.View.GONE
        }

        confidenceSlider.addOnChangeListener { _, value, _ ->
            confidenceThreshold = value
        }

        gpuSwitch.setOnCheckedChangeListener { _, isChecked ->
            initDetector(isChecked)
            // Reset average stats when hardware changes
            frameData.clear()
            avgFpsText.text = "Avg FPS: 0"
            avgLatencyText.text = "Avg Latency: 0ms"
        }

        cameraExecutor = Executors.newSingleThreadExecutor()

        // Check GPU compatibility and set default
        val compatList = CompatibilityList()
        val isGpuSupported = compatList.isDelegateSupportedOnThisDevice

        gpuSwitch.isChecked = isGpuSupported
        // Initialize Detector with GPU if supported, otherwise CPU
        initDetector(isGpuSupported)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun updateDebugInfo(inferenceTime: Long) {
        latencyText.text = "Latency: ${inferenceTime}ms"

        val currentTime = System.currentTimeMillis()

        // Add current frame data
        frameData.addLast(Pair(currentTime, inferenceTime))

        // Remove old data (older than 10 seconds)
        while (!frameData.isEmpty() && currentTime - frameData.peekFirst().first > 10000) {
            frameData.removeFirst()
        }

        // Calculate Average Latency (10s window)
        if (!frameData.isEmpty()) {
            var totalLatency = 0L
            for (p in frameData) {
                totalLatency += p.second
            }
            val avgLatency = totalLatency / frameData.size
            avgLatencyText.text = "Avg Latency: ${avgLatency}ms"
        }

        frameCount++
        val timeDiff = currentTime - lastFpsTimestamp

        // Update Instant FPS every 1 second (stats for last 1 sec)
        if (timeDiff >= 1000) {
            val fps = frameCount * 1000.0 / timeDiff
            fpsText.text = String.format("FPS: %.2f", fps)
            frameCount = 0
            lastFpsTimestamp = currentTime

            // Calculate Average FPS (10s window)
            // effective duration is min(10s, current duration of window)
            if (!frameData.isEmpty()) {
                val oldestTime = frameData.peekFirst()?.first
                val duration = currentTime - oldestTime!!
                // Avoid division by zero, though unlikely if list not empty and size > 1
                // If only 1 frame, duration is 0. 
                if (duration > 0) {
                    val avgFps = (frameData.size - 1) * 1000.0 / duration
                    // Note: strictly speaking, frames count is intervals. 
                    // If we have N frames, we have N-1 intervals. 
                    // For short duration, this is more accurate.
                    // Or for simple user facing "count over 10s": frameData.size / 10.0 (if full)

                    // Let's use simple count / window_size_seconds where window_size_seconds is bounded by 10.
                    // But if we just started, window size is small.

                    // Option A: frameData.size / ((currentTime - oldestTime)/1000.0)
                    // If window is full 10s, this is frameData.size / 10.

                    // val windowSizeSeconds = if (duration in 1..9999) duration / 1000.0 else 10.0
                    // If duration is very small (start), FPS might spike. 

                    // Let's stick to standard: (count) / (time_range)
                    val calculatedAvgFps =
                        frameData.size * 1000.0 / (if (duration < 100) 1000.0 else duration.toDouble())
                    avgFpsText.text = String.format("Avg FPS: %.2f", calculatedAvgFps)
                }
            }
        }
    }

    private fun initDetector(useGpu: Boolean) {
        cameraExecutor?.execute {
            val oldDetector = detector
            detector = null // Pause detection

            try {
                oldDetector?.close()
            } catch (e: Exception) {
                Log.e("MainActivity", "Error closing detector", e)
            }

            try {
                // 640
                // emulator: 5.45 fps, 180ms latency
                // S21U: 11 fps, 80ms latency
                // val modelName = "best_float32_640.tflite"

                // 512
                // emulator: 8.38 fps, 115ms latency
                // S21U: 13 fps, 70ms latency
                // val modelName = "best_float32_512.tflite"

                // 416
                // emulator: 12.04 fps, 75ms latency
                // S21U: 16 fps, 50ms latency
                // val modelName = "best_float32_416.tflite"

                // 320
                // emulator: 20.2 fps, 45ms latency
                // S21U: 21 fps, 40ms latency
                // val modelName = YoloDetector(this, "best_float32_320.tflite"

                // 640 half(fp16)
                // emulator: 5.3 fps, 184ms latency
                // S21U: 8.8 fps, 100ms latency
                //val modelName = "best_float16_640.tflite"

                // 512 half(fp16)
                // emulator: 8.2 fps, 118ms latency
                // S21U: 13 fps, 66ms latency
                // val modelName = "best_float16_512.tflite"

                // 448 half(fp16)
                // emulator: 8.2 fps, 118ms latency
                // S21U: 15 fps, 61ms latency
                val modelName = "best_float16_448.tflite"

                // 416 half(fp16)
                // emulator: 12 fps, 79ms latency
                // S21U: 16 fps, 51ms latency
                // val modelName = "best_float16_416.tflite"

                // 320 half(fp16)
                // emulator: 20.2 fps, 46.5ms latency
                // S21U: 23 fps, 30ms latency
                // val modelName = "best_float16_320.tflite"

                // 320 int8(quant, cpu)
                // emulator: 15+ fps, 45ms latency
                // S21U: 23 fps, 34ms latency
                // val modelName = "best_int8_320.tflite"

                val newDetector =
                    YoloDetector(this, modelName, useGpu = useGpu, labels = finetunedLabels)

                newDetector.setup()
                detector = newDetector

                runOnUiThread {
                    modelNameText.text = "Model: $modelName"
                }
            } catch (e: Exception) {
                Log.e("MainActivity", "Error initializing detector", e)
                runOnUiThread {
                    Toast.makeText(
                        this,
                        "Error initializing detector: ${e.message}",
                        Toast.LENGTH_LONG
                    ).show()

                    if (useGpu) {
                        Toast.makeText(
                            this,
                            "GPU init failed. Switching to CPU.",
                            Toast.LENGTH_SHORT
                        ).show()
                        gpuSwitch.isChecked = false
                        gpuSwitch.isEnabled = false
                    }
                }
            }
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build()
                .also {
                    it.surfaceProvider = viewFinder.surfaceProvider
                }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor!!) { image ->
                        processImage(image)
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )
            } catch (exc: Exception) {
                Log.e("MainActivity", "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun processImage(imageProxy: ImageProxy) {
        if (detector == null) {
            imageProxy.close()
            return
        }

        val bitmap = imageProxy.toBitmap()

        // Handle rotation if needed (toBitmap usually handles it if RGBA_8888 is used with latest CameraX, 
        // but sometimes we need to rotate manually based on imageProxy.imageInfo.rotationDegrees)
        // For now, let's assume toBitmap() gives us the correct orientation or we might need to rotate.
        // Actually, toBitmap() returns the bitmap as is in the buffer. We need to rotate it.

        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        val rotatedBitmap = if (rotationDegrees != 0) {
            rotateBitmap(bitmap, rotationDegrees.toFloat())
        } else {
            bitmap
        }

        val result = detector!!.detect(rotatedBitmap, confidenceThreshold)

        runOnUiThread {
            overlay.setInputImageSize(rotatedBitmap.width, rotatedBitmap.height)
            if (showBBoxOverlay && showDebugInfo) {
                overlay.setResults(result.boxes)
            } else {
                overlay.setResults(emptyList())
            }
            updateDebugInfo(result.inferenceTime)
        }

        imageProxy.close()
    }

    private fun rotateBitmap(bitmap: Bitmap, degrees: Float): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(degrees)
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor?.shutdown()
        detector?.close()
    }
}