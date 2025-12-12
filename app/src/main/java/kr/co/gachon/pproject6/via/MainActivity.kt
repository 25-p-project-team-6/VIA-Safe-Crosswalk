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
    private lateinit var zoomSwitch: android.widget.Switch
    private lateinit var debugContainer: android.widget.LinearLayout
    private lateinit var debugToggleButton: android.widget.ImageButton

    private var camera: androidx.camera.core.Camera? = null

    // Set this to false to hide debug info (FPS, Latency, Hardware, Slider)
    private var showDebugInfo = true

    // Set this to false to hide bounding boxes and labels
    private val showBBoxOverlay = true

    private var cameraExecutor: ExecutorService? = null

    @Volatile
    private var detector: YoloDetector? = null
    
    // User Settings
    private var generalObjThreshold = 0.5f // For non-traffic lights (Car, Bike, etc.)
    private var trafficLightThreshold = 0.15f // For Traffic Lights (Red, Green)
    
    // Global threshold passed to detector (min of the two)
    private var confidenceThreshold = 0.15f 
    
    private var currentModelName = "best_float16_640.tflite" // Default model

    private var lastFpsTimestamp = System.currentTimeMillis()
    private var frameCount = 0

    // Store timestamps and latencies for 10s sliding window
    // Pair(timestamp, latency)
    private val frameData = java.util.ArrayDeque<Pair<Long, Long>>()

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
        listOf("bicycle", "car", "motorcycle", "bus", "train", "truck", "green", "red")

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



        debugContainer.visibility =
            if (showDebugInfo) android.view.View.VISIBLE else android.view.View.GONE

        debugToggleButton.setOnClickListener {
            showDebugInfo = !showDebugInfo
            debugContainer.visibility =
                if (showDebugInfo) android.view.View.VISIBLE else android.view.View.GONE
        }

        // Initialize Slider Values
        confidenceSlider.value = 0.5f
        findViewById<com.google.android.material.slider.Slider>(R.id.trafficConfidenceSlider).value = 0.15f

        confidenceSlider.addOnChangeListener { _, value, _ ->
            generalObjThreshold = value
            findViewById<android.widget.TextView>(R.id.confidenceSliderLabel).text =
                String.format("General Confidence: %.2f", value)
            updateDetectorThresholds()
        }

        findViewById<com.google.android.material.slider.Slider>(R.id.trafficConfidenceSlider).addOnChangeListener { _, value, _ ->
            trafficLightThreshold = value
            findViewById<android.widget.TextView>(R.id.trafficConfidenceLabel).text =
                String.format("Traffic Confidence: %.2f", value)
            updateDetectorThresholds()
        }

        gpuSwitch.setOnCheckedChangeListener { _, isChecked ->
            initDetector(isChecked)
            // Reset average stats when hardware changes
            frameData.clear()
            avgFpsText.text = "Avg FPS: 0"
            avgLatencyText.text = "Avg Latency: 0ms"
        }

        zoomSwitch = findViewById(R.id.swZoom2x)
        zoomSwitch.setOnCheckedChangeListener { _, isChecked ->
            val zoomRatio = if (isChecked) 2.0f else 1.0f
            camera?.cameraControl?.setZoomRatio(zoomRatio)
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

        setupModelSpinner()
    }

    private fun updateDetectorThresholds() {
        // Set global threshold to General Slider value (fallback)
        confidenceThreshold = generalObjThreshold
        
        // Explicitly map each class to its respective slider value
        val specificMap = mutableMapOf<String, Float>()
        
        // 1. Traffic Lights -> Traffic Slider
        specificMap["green"] = trafficLightThreshold
        specificMap["red"] = trafficLightThreshold
        
        // 2. Verified Objects -> General Slider
        val others = listOf("bicycle", "car", "motorcycle", "bus", "train", "truck")
        for (label in others) {
            specificMap[label] = generalObjThreshold
        }
        
        detector?.specificConfidenceThresholds = specificMap
    }

    private fun setupModelSpinner() {
        val spinner = findViewById<android.widget.Spinner>(R.id.modelSpinner)
        try {
            // Scan assets for .tflite files
            val assetManager = assets
            val files = assetManager.list("")
            val modelFiles = files?.filter { it.endsWith(".tflite") }?.sorted() ?: emptyList()

            if (modelFiles.isNotEmpty()) {
                val adapter = android.widget.ArrayAdapter(
                    this,
                    android.R.layout.simple_spinner_item,
                    modelFiles
                )
                adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
                spinner.adapter = adapter

                // Set selection to current default if exists
                val defaultIndex = modelFiles.indexOf(currentModelName)
                if (defaultIndex >= 0) {
                    spinner.setSelection(defaultIndex)
                }

                spinner.onItemSelectedListener =
                    object : android.widget.AdapterView.OnItemSelectedListener {
                        override fun onItemSelected(
                            parent: android.widget.AdapterView<*>?,
                            view: android.view.View?,
                            position: Int,
                            id: Long
                        ) {
                            val selectedModel = modelFiles[position]
                            if (selectedModel != currentModelName) {
                                currentModelName = selectedModel
                                // Re-init detector with new model
                                initDetector(gpuSwitch.isChecked)
                            }
                        }

                        override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {}
                    }
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "Error setting up model spinner", e)
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

                if (duration > 0) {
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
                // 448 half(fp16)
                val modelName = currentModelName

                val newDetector = YoloDetector(
                    this,
                    modelName,
                    useGpu = useGpu,
                    labels = finetunedLabels,
                    defaultIouThreshold = 0.5f,
                    specificIouThresholds = mapOf("red" to 0.05f, "green" to 0.05f)
                )

                newDetector.setup()
                detector = newDetector
                
                // Apply current user settings
                updateDetectorThresholds()

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
                camera = cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )

                // Check for 2x Zoom support and auto-enable
                val zoomState = camera?.cameraInfo?.zoomState?.value
                if (zoomState != null) {
                    val maxZoom = zoomState.maxZoomRatio
                    if (maxZoom >= 2.0f) {
                        runOnUiThread {
                            if (!zoomSwitch.isChecked) {
                                zoomSwitch.isChecked =
                                    true // This will trigger listener and set zoom to 2.0
                                // If listener doesn't trigger automatically on setChecked (sometimes it doesn't if not attached), force it
                                camera?.cameraControl?.setZoomRatio(2.0f)
                            }
                        }
                    } else {
                        runOnUiThread {
                            zoomSwitch.isEnabled = false
                            zoomSwitch.text = "2x Zoom (Not Supported)"
                        }
                    }
                }

                // Also observe zoom state for dynamic updates if needed
                camera?.cameraInfo?.zoomState?.observe(this) { state ->
                    // Just log or update UI if needed
                    // Log.d("MainActivity", "Zoom: ${state.zoomRatio}x / Max: ${state.maxZoomRatio}x")
                }

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

        // Execute PostProcessor for logging (logic test)
        // This will print logs but we don't use the return value for the overlay
        var trafficState = PostProcessor.TrafficLightState.UNKNOWN

        // Check switches (UI thread check not ideal here but safe enough if cached)
        val showRawBoxes =
            findViewById<android.widget.Switch>(R.id.swRawDetection)?.isChecked == true
        val enableTrafficLogic =
            findViewById<android.widget.Switch>(R.id.swTrafficLogic)?.isChecked == true
        val enableHighlight =
            findViewById<android.widget.Switch>(R.id.swHighlightTarget)?.isChecked == true

        var targetScore = 0f
        var targetCls = "None"

        var targetBox: OverlayView.BoundingBox? = null
        // Keep track of boxes to show. Default to raw result.
        var boxesToShow = result.boxes

        if (enableTrafficLogic) {
            // Execute PostProcessor for logging and logic
            val correctedBoxes = PostProcessor.applyColorCorrection(rotatedBitmap, result.boxes)
            val targetData = PostProcessor.selectTargetTrafficLight(correctedBoxes)

            targetBox = targetData?.first
            targetScore = targetData?.second ?: 0f
            if (targetBox != null) {
                targetCls = targetBox.clsName
                if (enableHighlight) {
                    targetBox.isTarget = true // Highlight the target only if enabled
                }
            }

            trafficState = PostProcessor.updateTrafficLightState(targetBox)
            boxesToShow = correctedBoxes // Show processed boxes (with flags/color swaps)
        }

        runOnUiThread {
            overlay.setInputImageSize(rotatedBitmap.width, rotatedBitmap.height)

            // Logic for Overlay Visibility
            // If showRawBoxes is ON, show boxesToShow (which is either raw or corrected based on logic switch)
            // If showRawBoxes is OFF, show nothing.
            if (showRawBoxes && showBBoxOverlay) { // showBBoxOverlay is controlled by eye icon
                overlay.setResults(boxesToShow)
            } else {
                overlay.setResults(emptyList())
            }
            updateDebugInfo(result.inferenceTime)

            // Update Target Info TextView
            val targetText = findViewById<android.widget.TextView>(R.id.targetInfoText)
            if (targetText != null) {
                targetText.text = if (enableTrafficLogic) {
                    val ratioText = if (targetBox != null && targetBox.debugRatio >= 0) {
                        String.format(" (Ratio: %.2f)", targetBox.debugRatio)
                    } else {
                        ""
                    }
                    String.format("Target: %s (Score: %.5f)%s", targetCls, targetScore, ratioText)
                } else {
                    "Logic Disabled"
                }
            }

            // Update Border UI
            val statusBorder = findViewById<android.view.View>(R.id.statusBorder)
            if (enableTrafficLogic) {
                when (trafficState) {
                    PostProcessor.TrafficLightState.RED -> statusBorder.setBackgroundResource(R.drawable.border_red)
                    PostProcessor.TrafficLightState.GREEN -> statusBorder.setBackgroundResource(R.drawable.border_green)
                    else -> statusBorder.setBackgroundResource(R.drawable.border_transparent)
                }
            } else {
                statusBorder.setBackgroundResource(R.drawable.border_transparent)
            }
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