package kr.co.gachon.pproject6.via

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.ImageProxy
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.activity.result.contract.ActivityResultContracts
import com.google.android.material.slider.Slider
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import org.tensorflow.lite.gpu.CompatibilityList
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat

import kr.co.gachon.pproject6.via.camera.CameraManager
import kr.co.gachon.pproject6.via.ml.PostProcessor
import kr.co.gachon.pproject6.via.ml.YoloDetector
import kr.co.gachon.pproject6.via.ml.ObjectTracker
import kr.co.gachon.pproject6.via.ui.OverlayView
import kr.co.gachon.pproject6.via.util.ImageUtils
import kr.co.gachon.pproject6.via.util.PerformanceTracker

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

    private var cameraManager: CameraManager? = null

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

    // GPU Support Flag
    private var isGpuSupported = false
    
    private var currentModelName = "best_float16_640.tflite" // Default model
    
    // Performance Tracker
    private val performanceTracker = PerformanceTracker()

    // Object Tracker
    private val objectTracker = ObjectTracker()

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

        // Initialize Slider Values (Trigger listeners)
        confidenceSlider.value = 0.5f
        findViewById<com.google.android.material.slider.Slider>(R.id.trafficConfidenceSlider).value = 0.15f

        gpuSwitch.setOnCheckedChangeListener { _, isChecked ->
            initDetector(isChecked)
            // Reset average stats when hardware changes
            performanceTracker.clear()
            avgFpsText.text = "Avg FPS: 0"
            avgLatencyText.text = "Avg Latency: 0ms"
        }

        zoomSwitch = findViewById(R.id.swZoom2x)
        zoomSwitch.setOnCheckedChangeListener { _, isChecked ->
            val zoomRatio = if (isChecked) 2.0f else 1.0f
            cameraManager?.setZoom(zoomRatio)
        }

        cameraExecutor = Executors.newSingleThreadExecutor()

        // Check GPU compatibility and set default
        val compatList = CompatibilityList()
        isGpuSupported = compatList.isDelegateSupportedOnThisDevice

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
                                
                                // Reset logic: Update switch state availability based on hardware support
                                runOnUiThread {
                                    if (isGpuSupported) {
                                        gpuSwitch.isEnabled = true
                                        // Auto-turn ON GPU if supported
                                        if (!gpuSwitch.isChecked) {
                                            gpuSwitch.isChecked = true // Triggers listener -> initDetector(true)
                                        } else {
                                            // Already On, listener won't fire, so init manually
                                            initDetector(true)
                                        }
                                    } else {
                                        // GPU not supported by device
                                        initDetector(false)
                                    }
                                }
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
        
        // Delegate calculation to Tracker
        performanceTracker.update(inferenceTime)
        
        // Update UI with results
        fpsText.text = performanceTracker.currentFpsStr
        avgFpsText.text = performanceTracker.avgFpsStr
        avgLatencyText.text = performanceTracker.avgLatencyStr
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
        cameraManager = CameraManager(this, this, viewFinder, cameraExecutor!!) { image ->
            processImage(image)
        }
        
        cameraManager?.startCamera { maxZoom ->
            // Update Zoom Switch UI based on supported Max Zoom
            if (maxZoom >= 2.0f) {
                runOnUiThread {
                    if (!zoomSwitch.isChecked) {
                        zoomSwitch.isChecked = true 
                        // Force update via manager
                        cameraManager?.setZoom(2.0f)
                    }
                }
            } else {
                runOnUiThread {
                    zoomSwitch.isEnabled = false
                    zoomSwitch.text = "2x Zoom (Not Supported)"
                }
            }
        }
    }
    
    private fun processImage(imageProxy: ImageProxy) {
        if (detector == null) {
            imageProxy.close()
            return
        }

        val bitmap = imageProxy.toBitmap()
        
        // Use ImageUtils for rotation
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        val rotatedBitmap = ImageUtils.rotateBitmap(bitmap, rotationDegrees.toFloat())

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
            
            // Use ObjectTracker to select target (with tracking)
            val targetData = objectTracker.selectTarget(correctedBoxes)

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
                    String.format("Target: %s (Score: %.2f)%s", targetCls, targetScore, ratioText)
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

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor?.shutdown()
        detector?.close()
    }
}
