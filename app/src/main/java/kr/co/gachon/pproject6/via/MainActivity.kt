package kr.co.gachon.pproject6.via

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.RectF
import android.location.Location
import android.location.LocationManager
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.ImageProxy
import androidx.camera.view.PreviewView
import androidx.camera.view.transform.CoordinateTransform
import androidx.camera.view.transform.ImageProxyTransformFactory
import androidx.camera.view.transform.OutputTransform
import com.google.android.material.button.MaterialButton
import com.google.android.material.slider.Slider
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.content.ContextCompat
import kr.co.gachon.pproject6.via.feedback.SignalFeedbackManager
import kr.co.gachon.pproject6.via.camera.CameraManager
import kr.co.gachon.pproject6.via.context.CrossingSupportManager
import kr.co.gachon.pproject6.via.context.CrossingSupportSnapshot
import kr.co.gachon.pproject6.via.ml.GuidanceBlockReason
import kr.co.gachon.pproject6.via.ml.GuidancePhase
import kr.co.gachon.pproject6.via.ml.GuidanceStateStabilizer
import kr.co.gachon.pproject6.via.ml.GuidanceTuningDefaults
import kr.co.gachon.pproject6.via.ml.InferenceModelProfile
import kr.co.gachon.pproject6.via.ml.PostProcessor
import kr.co.gachon.pproject6.via.ml.SignalAnalysisResult
import kr.co.gachon.pproject6.via.ml.SignalAnalyzer
import kr.co.gachon.pproject6.via.ml.TrafficLightState
import kr.co.gachon.pproject6.via.ml.UserGuidanceState
import kr.co.gachon.pproject6.via.ml.YoloDetector
import kr.co.gachon.pproject6.via.ml.toGuidanceSnapshot
import kr.co.gachon.pproject6.via.ml.withGuidanceSnapshot
import kr.co.gachon.pproject6.via.map.KineticGuestSessionManager
import kr.co.gachon.pproject6.via.map.MapDebugCacheManager
import kr.co.gachon.pproject6.via.onboarding.AppPreferences
import kr.co.gachon.pproject6.via.onboarding.OnboardingActivity
import kr.co.gachon.pproject6.via.ui.OverlayView
import kr.co.gachon.pproject6.via.util.ImageUtils
import kr.co.gachon.pproject6.via.util.PerformanceTracker
import kr.co.gachon.pproject6.via.util.RateTracker
import org.tensorflow.lite.gpu.CompatibilityList
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

class MainActivity : AppCompatActivity() {
    private companion object {
        private const val CAMERA_COLD_START_TIMEOUT_MS = 3_500L
        private const val MAX_CAMERA_COLD_START_RECOVERIES = 1
    }

    private lateinit var viewFinder: PreviewView
    private lateinit var overlay: OverlayView
    private lateinit var fpsText: TextView
    private lateinit var avgFpsText: TextView
    private lateinit var latencyText: TextView
    private lateinit var avgLatencyText: TextView
    private lateinit var stageBreakdownText: TextView
    private lateinit var backendStatusText: TextView
    private lateinit var inputFpsText: TextView
    private lateinit var modelNameText: TextView
    private lateinit var buildInfoText: TextView
    private lateinit var resetAppButton: MaterialButton
    private lateinit var openGpsDebugMapButton: MaterialButton
    private lateinit var clearMapCacheButton: MaterialButton
    private lateinit var targetInfoText: TextView
    private lateinit var decisionDebugText: TextView
    private lateinit var tuningDebugText: TextView
    private lateinit var statusTitleText: TextView
    private lateinit var statusDetailText: TextView
    private lateinit var confidenceSliderLabel: TextView
    private lateinit var trafficConfidenceLabel: TextView
    private lateinit var downTiltLabel: TextView
    private lateinit var confidenceSlider: Slider
    private lateinit var trafficConfidenceSlider: Slider
    private lateinit var downTiltSlider: Slider
    private lateinit var gpuSwitch: com.google.android.material.switchmaterial.SwitchMaterial
    private lateinit var zoomSwitch: androidx.appcompat.widget.SwitchCompat
    private lateinit var rawDetectionSwitch: androidx.appcompat.widget.SwitchCompat
    private lateinit var trafficLogicSwitch: androidx.appcompat.widget.SwitchCompat
    private lateinit var highlightTargetSwitch: androidx.appcompat.widget.SwitchCompat
    private lateinit var debugContainer: View
    private lateinit var debugToggleButton: android.widget.ImageButton
    private lateinit var buildInfoCard: View
    private lateinit var topControlCard: View
    private lateinit var statusPanel: View
    private lateinit var statusBorder: View
    private var lastLoggedDecisionSummary: String? = null
    private var lastLoggedMapSummary: String? = null
    private var suppressGpuToggleCallback = false
    private var suppressZoomToggleCallback = false
    private lateinit var preferences: AppPreferences
    private val locationManager by lazy {
        getSystemService(LocationManager::class.java)
    }
    private var latestCrossingSupportSnapshot: CrossingSupportSnapshot = CrossingSupportSnapshot()

    private var cameraManager: CameraManager? = null
    private var hasStartedCamera = false
    private var cameraRecoveryAttempts = 0
    @Volatile
    private var lastCameraFrameAtElapsedMs = 0L

    // Set this to false to hide debug info (FPS, Latency, Hardware, Slider)
    private var showDebugInfo = false

    // Set this to false to hide bounding boxes and labels
    private val showBBoxOverlay = true

    private var cameraExecutor: ExecutorService? = null
    private var processingExecutor: ExecutorService? = null

    @Volatile
    private var detector: YoloDetector? = null
    
    // User Settings
    private var generalObjThreshold = 0.5f // For non-traffic lights (Car, Bike, etc.)
    private var trafficLightThreshold = 0.15f // For Traffic Lights (Red, Green)
    
    // Global threshold passed to detector (min of the two)
    private var confidenceThreshold = 0.15f 

    // GPU Support Flag
    private var isGpuSupported = false
    
    private var currentModelName = "best_float16_640.tflite" // Practical default for current device targets
    private var currentModelProfile = InferenceModelProfile.fromFileName(currentModelName)
    private var availableModelFiles: List<String> = emptyList()
    private var initialBackendPreference: String? = null
    private var reusableRotatedBitmap: Bitmap? = null
    
    // Performance Tracker
    private val performanceTracker = PerformanceTracker()
    private val cameraRateTracker = RateTracker(label = "Camera FPS")
    private val pendingFrame = AtomicReference<ImageProxy?>(null)
    private val processingScheduled = AtomicBoolean(false)
    private val imageProxyTransformFactory =
        ImageProxyTransformFactory().apply {
            setUsingCropRect(true)
            setUsingRotationDegrees(true)
        }

    private val signalAnalyzer = SignalAnalyzer()
    private val guidanceStateStabilizer =
        GuidanceStateStabilizer(GuidanceTuningDefaults.guidanceStabilizerConfig)
    private lateinit var crossingSupportManager: CrossingSupportManager
    private lateinit var feedbackManager: SignalFeedbackManager
    private val guidanceRuntimeResetter by lazy {
        GuidanceRuntimeResetter(
            resetAnalyzer = { signalAnalyzer.reset() },
            resetStabilizer = { guidanceStateStabilizer.reset() },
            resetCrossingSupport = { crossingSupportManager.reset() },
            clearFeedback = { feedbackManager.clearState() },
            afterReset = {
                lastLoggedDecisionSummary = null
                lastLoggedMapSummary = null
                latestCrossingSupportSnapshot = CrossingSupportSnapshot()
            }
        )
    }

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
        preferences = AppPreferences(this)
        if (!preferences.onboardingCompleted) {
            startActivity(Intent(this, OnboardingActivity::class.java))
            finish()
            return
        }

        preferences.selectedModelName?.let { savedModel ->
            currentModelName = savedModel
            currentModelProfile = InferenceModelProfile.fromFileName(savedModel)
        }
        initialBackendPreference = preferences.selectedBackendLabel
        KineticGuestSessionManager.from(this).prefetchIfNeeded()

        setContentView(R.layout.activity_main)

        // Handle Window Insets for edge-to-edge
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        viewFinder = findViewById(R.id.viewFinder)
        viewFinder.implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        viewFinder.scaleType = PreviewView.ScaleType.FILL_CENTER
        overlay = findViewById(R.id.overlay)
        debugContainer = findViewById(R.id.debugContainer)
        debugToggleButton = findViewById(R.id.debugToggleButton)
        buildInfoCard = findViewById(R.id.buildInfoCard)
        topControlCard = findViewById(R.id.topControlCard)
        statusPanel = findViewById(R.id.statusPanel)
        backendStatusText = findViewById(R.id.backendStatusText)
        resetAppButton = findViewById(R.id.resetAppButton)
        openGpsDebugMapButton = findViewById(R.id.openGpsDebugMapButton)
        clearMapCacheButton = findViewById(R.id.clearMapCacheButton)
        inputFpsText = findViewById(R.id.inputFpsText)
        modelNameText = findViewById(R.id.modelNameText)
        buildInfoText = findViewById(R.id.buildInfoText)
        fpsText = findViewById(R.id.fpsText)
        avgFpsText = findViewById(R.id.avgFpsText)
        latencyText = findViewById(R.id.latencyText)
        avgLatencyText = findViewById(R.id.avgLatencyText)
        stageBreakdownText = findViewById(R.id.stageBreakdownText)
        targetInfoText = findViewById(R.id.targetInfoText)
        decisionDebugText = findViewById(R.id.decisionDebugText)
        tuningDebugText = findViewById(R.id.tuningDebugText)
        statusTitleText = findViewById(R.id.statusTitleText)
        statusDetailText = findViewById(R.id.statusDetailText)
        confidenceSliderLabel = findViewById(R.id.confidenceSliderLabel)
        trafficConfidenceLabel = findViewById(R.id.trafficConfidenceLabel)
        downTiltLabel = findViewById(R.id.downTiltLabel)
        confidenceSlider = findViewById(R.id.confidenceSlider)
        trafficConfidenceSlider = findViewById(R.id.trafficConfidenceSlider)
        downTiltSlider = findViewById(R.id.downTiltSlider)
        gpuSwitch = findViewById(R.id.gpuSwitch)
        zoomSwitch = findViewById(R.id.swZoom2x)
        rawDetectionSwitch = findViewById(R.id.swRawDetection)
        trafficLogicSwitch = findViewById(R.id.swTrafficLogic)
        highlightTargetSwitch = findViewById(R.id.swHighlightTarget)
        statusBorder = findViewById(R.id.statusBorder)
        feedbackManager = SignalFeedbackManager(this)
        crossingSupportManager = CrossingSupportManager(this, GuidanceTuningDefaults.crossingSupportConfig)
        buildInfoText.text = "v${BuildConfig.VERSION_NAME} (${BuildConfig.VERSION_CODE}) · ${BuildConfig.BUILD_STAMP}"
        updateTuningDebugText()
        Log.i("VIA_GUIDANCE", "tuning=${GuidanceTuningDefaults.toDebugSummary()}")
        modelNameText.text = "모델: ${currentModelProfile.displayNameWithSize()}"

        debugContainer.visibility =
            if (showDebugInfo) View.VISIBLE else View.GONE

        applySystemBarInsets()

        debugToggleButton.setOnClickListener {
            showDebugInfo = !showDebugInfo
            debugContainer.visibility =
                if (showDebugInfo) View.VISIBLE else View.GONE
            debugToggleButton.contentDescription =
                if (showDebugInfo) "디버그 정보 닫기" else "디버그 정보 열기"
        }
        resetAppButton.setOnClickListener {
            detector?.close()
            detector = null
            cameraManager?.stopCamera()
            preferences.clearCalibration()
            val intent = Intent(this, OnboardingActivity::class.java).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
            }
            startActivity(intent)
            finishAffinity()
            finish()
        }
        openGpsDebugMapButton.setOnClickListener {
            openGpsDebugMap()
        }
        clearMapCacheButton.setOnClickListener {
            clearMapCaches()
        }
        updateGpsDebugMapButtonState()

        confidenceSlider.addOnChangeListener { _, value, _ ->
            generalObjThreshold = value
            confidenceSliderLabel.text = String.format("General Confidence: %.2f", value)
            updateDetectorThresholds()
        }

        trafficConfidenceSlider.addOnChangeListener { _, value, _ ->
            trafficLightThreshold = value
            trafficConfidenceLabel.text = String.format("Traffic Confidence: %.2f", value)
            updateDetectorThresholds()
        }

        downTiltSlider.addOnChangeListener { _, value, _ ->
            crossingSupportManager.updateLookingDownThresholdDegrees(value)
            updateDownTiltLabel()
            updateTuningDebugText()
        }

        confidenceSlider.value = 0.5f
        trafficConfidenceSlider.value = 0.15f
        downTiltSlider.value = 20f
        downTiltSlider.isEnabled = false
        updateDownTiltLabel()

        gpuSwitch.setOnCheckedChangeListener { _, isChecked ->
            if (suppressGpuToggleCallback) {
                return@setOnCheckedChangeListener
            }
            initDetector(isChecked, restartCameraAfterInit = hasStartedCamera)
            resetPerformanceStats()
        }

        zoomSwitch.setOnCheckedChangeListener { _, isChecked ->
            if (suppressZoomToggleCallback) {
                return@setOnCheckedChangeListener
            }
            applySelectedZoom()
        }
        trafficLogicSwitch.setOnCheckedChangeListener { _, isChecked ->
            if (!isChecked) {
                guidanceRuntimeResetter.resetForTrafficLogicDisabled()
            }
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
        processingExecutor = Executors.newSingleThreadExecutor()

        // Check GPU compatibility, but still try GPU first for deployable GPU-friendly models.
        val compatList = CompatibilityList()
        isGpuSupported = compatList.isDelegateSupportedOnThisDevice

        availableModelFiles = discoverModelFiles()
        InferenceModelProfile.recommend(availableModelFiles, isGpuSupported)?.let { recommendedProfile ->
            if (preferences.selectedModelName == null || preferences.selectedModelName !in availableModelFiles) {
                currentModelName = recommendedProfile.fileName
                currentModelProfile = recommendedProfile
            }
        }
        val shouldUseSavedBackend =
            initialBackendPreference?.contains("GPU") == true &&
                currentModelProfile.recommendedUseGpu
        configureGpuSwitch(
            checked = shouldUseSavedBackend || currentModelProfile.recommendedUseGpu,
            enabled = currentModelProfile.recommendedUseGpu
        )
        publishBackendStatus("Initializing…")
        // Initialize detector with the recommended delegate for the selected model.
        initDetector(gpuSwitch.isChecked, restartCameraAfterInit = false)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }

        setupModelSpinner()
    }

    override fun onResume() {
        super.onResume()
        crossingSupportManager.start()
        latestCrossingSupportSnapshot = crossingSupportManager.snapshot()
        updateGpsDebugMapButtonState()
    }

    override fun onPause() {
        guidanceRuntimeResetter.resetForPause()
        viewFinder.removeCallbacks(cameraColdStartRecoveryRunnable)
        crossingSupportManager.stop()
        super.onPause()
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

    private fun discoverModelFiles(): List<String> {
        return try {
            assets.list("")
                ?.filter { it.endsWith(".tflite", ignoreCase = true) }
                ?.sorted()
                ?: emptyList()
        } catch (e: Exception) {
            Log.e("MainActivity", "Error scanning model assets", e)
            emptyList()
        }
    }

    private fun configureGpuSwitch(checked: Boolean, enabled: Boolean) {
        suppressGpuToggleCallback = true
        gpuSwitch.isChecked = checked
        gpuSwitch.isEnabled = enabled
        gpuSwitch.text = when {
            !currentModelProfile.recommendedUseGpu -> "Use GPU (INT8 uses CPU)"
            else -> "Use GPU"
        }
        suppressGpuToggleCallback = false
    }

    private fun applyModelSelection(selectedModel: String) {
        currentModelName = selectedModel
        currentModelProfile = InferenceModelProfile.fromFileName(selectedModel)
        reusableRotatedBitmap = null

        val shouldEnableGpu = currentModelProfile.recommendedUseGpu
        configureGpuSwitch(
            checked = shouldEnableGpu,
            enabled = shouldEnableGpu
        )

        resetPerformanceStats()
        initDetector(gpuSwitch.isChecked, restartCameraAfterInit = hasStartedCamera)
    }

    private fun setupModelSpinner() {
        val spinner = findViewById<android.widget.Spinner>(R.id.modelSpinner)
        try {
            val modelFiles = availableModelFiles

            if (modelFiles.isNotEmpty()) {
                val modelProfiles = modelFiles.map(InferenceModelProfile::fromFileName)
                val adapter = android.widget.ArrayAdapter(
                    this,
                    android.R.layout.simple_spinner_item,
                    modelProfiles.map { it.displayNameWithSize() }
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
                            val selectedModel = modelProfiles[position].fileName
                            if (selectedModel != currentModelName) {
                                applyModelSelection(selectedModel)
                            }
                        }

                        override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {}
                    }
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "Error setting up model spinner", e)
        }
    }

    private fun updateTuningDebugText() {
        tuningDebugText.text =
            "Tuning: ${GuidanceTuningDefaults.toDebugSummary()}, tilt raw down=-160..-90, up=90..120"
    }

    private fun updateDownTiltLabel() {
        downTiltLabel.text = "Tilt Raw Range: down -160..-90 / up 90..120"
    }

    private fun updateDebugInfo(
        inferenceTime: Long,
        totalLatencyMs: Long,
        stageDurationsMs: Map<String, Long>
    ) {
        latencyText.text = "Detect: ${inferenceTime}ms | Total: ${totalLatencyMs}ms"
        
        // Delegate calculation to Tracker
        performanceTracker.update(inferenceTime, stageDurationsMs)
        
        // Update UI with results
        inputFpsText.text = cameraRateTracker.rateStr
        fpsText.text = performanceTracker.currentFpsStr.replaceFirst("FPS", "Processed FPS")
        avgFpsText.text = performanceTracker.avgFpsStr.replaceFirst("Avg FPS", "Processed Avg FPS")
        avgLatencyText.text = performanceTracker.avgLatencyStr
        stageBreakdownText.text = performanceTracker.stageBreakdownStr
    }

    private fun initDetector(
        useGpu: Boolean,
        restartCameraAfterInit: Boolean
    ) {
        val executor = processingExecutor ?: return
        if (restartCameraAfterInit) {
            runOnUiThread {
                cameraManager?.stopCamera()
            }
            pendingFrame.getAndSet(null)?.close()
        }
        executor.execute {
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
                    val backendStatus =
                        "model=$modelName backend=${newDetector.runtimeBackendLabel} " +
                            "requestedGpu=$useGpu compat=${newDetector.compatibilityReportedSupported} " +
                            "analysis=${currentModelProfile.analysisResolution.width}x" +
                            "${currentModelProfile.analysisResolution.height}"
                    modelNameText.text =
                        "모델: ${currentModelProfile.displayNameWithSize()} (${newDetector.runtimeBackendLabel})"
                    Log.i("VIA_GPU", backendStatus)
                    publishBackendStatus("Backend: ${newDetector.runtimeBackendLabel}")
                    if (restartCameraAfterInit && hasStartedCamera) {
                        startCamera()
                    }
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
                        publishBackendStatus("Backend: CPU")
                        configureGpuSwitch(
                            checked = false,
                            enabled = currentModelProfile.recommendedUseGpu
                        )
                        initDetector(false, restartCameraAfterInit = restartCameraAfterInit)
                    }
                }
            }
        }
    }

    private fun startCamera(isRecoveryRestart: Boolean = false) {
        cameraManager?.stopCamera()
        hasStartedCamera = true
        if (!isRecoveryRestart) {
            cameraRecoveryAttempts = 0
        }
        lastCameraFrameAtElapsedMs = 0L
        val resolution = currentModelProfile.analysisResolution
        cameraManager = CameraManager(
            this,
            this,
            viewFinder,
            cameraExecutor!!,
            android.util.Size(resolution.width, resolution.height)
        ) { image ->
            enqueueFrame(image)
        }

        runOnUiThread {
            zoomSwitch.isEnabled = false
            zoomSwitch.text = "2x Zoom 확인 중"
        }
        
        cameraManager?.startCamera { maxZoom ->
            // Update Zoom Switch UI based on supported Max Zoom
            if (maxZoom >= 2.0f) {
                runOnUiThread {
                    zoomSwitch.isEnabled = true
                    zoomSwitch.text = "Use 2x Zoom"
                    applySelectedZoom()
                }
            } else {
                runOnUiThread {
                    suppressZoomToggleCallback = true
                    zoomSwitch.isChecked = false
                    suppressZoomToggleCallback = false
                    zoomSwitch.isEnabled = false
                    zoomSwitch.text = "2x Zoom (Not Supported)"
                    applySelectedZoom()
                }
            }
        }
        scheduleCameraColdStartRecovery()
    }

    private fun applySelectedZoom() {
        val zoomRatio = if (zoomSwitch.isChecked && zoomSwitch.isEnabled) 2.0f else 1.0f
        cameraManager?.setZoom(zoomRatio)
    }
    
    private fun enqueueFrame(imageProxy: ImageProxy) {
        lastCameraFrameAtElapsedMs = SystemClock.elapsedRealtime()
        cameraRateTracker.mark()
        val previousFrame = pendingFrame.getAndSet(imageProxy)
        previousFrame?.close()
        scheduleFrameProcessing()
    }

    private fun scheduleCameraColdStartRecovery() {
        viewFinder.removeCallbacks(cameraColdStartRecoveryRunnable)
        viewFinder.postDelayed(cameraColdStartRecoveryRunnable, CAMERA_COLD_START_TIMEOUT_MS)
    }

    private val cameraColdStartRecoveryRunnable =
        Runnable {
            if (!hasStartedCamera) {
                return@Runnable
            }
            if (cameraRecoveryAttempts >= MAX_CAMERA_COLD_START_RECOVERIES) {
                return@Runnable
            }
            val lastFrameAt = lastCameraFrameAtElapsedMs
            val stalled =
                lastFrameAt == 0L ||
                    (SystemClock.elapsedRealtime() - lastFrameAt) >= CAMERA_COLD_START_TIMEOUT_MS
            if (!stalled) {
                return@Runnable
            }
            cameraRecoveryAttempts += 1
            Log.w(
                "MainActivity",
                "No camera frames detected after startup, restarting preview (attempt=$cameraRecoveryAttempts)"
            )
            publishBackendStatus("카메라 초기화 재시도 중…")
            startCamera(isRecoveryRestart = true)
        }

    private fun scheduleFrameProcessing() {
        val executor = processingExecutor ?: run {
            pendingFrame.getAndSet(null)?.close()
            return
        }
        if (!processingScheduled.compareAndSet(false, true)) {
            return
        }

        executor.execute {
            try {
                while (true) {
                    val nextFrame = pendingFrame.getAndSet(null) ?: break
                    processImage(nextFrame)
                }
            } finally {
                processingScheduled.set(false)
                if (pendingFrame.get() != null) {
                    scheduleFrameProcessing()
                }
            }
        }
    }

    private fun processImage(imageProxy: ImageProxy) {
        val activeDetector = detector
        if (activeDetector == null) {
            imageProxy.close()
            return
        }

        val stageDurationsMs = linkedMapOf<String, Long>()
        val frameStartNs = SystemClock.elapsedRealtimeNanos()
        var imageClosed = false

        try {
            val copyStartNs = SystemClock.elapsedRealtimeNanos()
            val analysisOutputTransform = imageProxyTransformFactory.getOutputTransform(imageProxy)
            val cropRect = imageProxy.cropRect
            val bitmap = imageProxy.toBitmap()
            val rotationDegrees = imageProxy.imageInfo.rotationDegrees
            imageProxy.close()
            imageClosed = true
            stageDurationsMs["copy"] = elapsedMillis(copyStartNs)

            val rotateStartNs = SystemClock.elapsedRealtimeNanos()
            val croppedBitmap = ImageUtils.cropBitmap(bitmap, cropRect)
            val rotatedBitmap = ImageUtils.rotateBitmap(
                bitmap = croppedBitmap,
                degrees = rotationDegrees.toFloat(),
                reusableBitmap = reusableRotatedBitmap
            )
            if (rotatedBitmap !== croppedBitmap) {
                reusableRotatedBitmap = rotatedBitmap
            }
            stageDurationsMs["rotate"] = elapsedMillis(rotateStartNs)

            val detectStartNs = SystemClock.elapsedRealtimeNanos()
            val result = activeDetector.detect(rotatedBitmap, confidenceThreshold)
            stageDurationsMs["detect"] = elapsedMillis(detectStartNs)

            val enableTrafficLogic = trafficLogicSwitch.isChecked
            val crossingSupportSnapshot =
                if (enableTrafficLogic) {
                    crossingSupportManager.snapshot()
                } else {
                    CrossingSupportSnapshot()
                }
            val analyzeStartNs = SystemClock.elapsedRealtimeNanos()
            val rawAnalysisResult = signalAnalyzer.analyze(
                bitmap = rotatedBitmap,
                rawBoxes = result.boxes,
                enableTrafficLogic = enableTrafficLogic,
                enableHighlight = highlightTargetSwitch.isChecked,
                crossingSupportSnapshot = crossingSupportSnapshot
            )
            val analysisResult =
                if (enableTrafficLogic) {
                    rawAnalysisResult.withGuidanceSnapshot(
                        guidanceStateStabilizer.stabilize(rawAnalysisResult.toGuidanceSnapshot())
                    )
                } else {
                    guidanceStateStabilizer.reset()
                    rawAnalysisResult
                }
            stageDurationsMs["analyze"] = elapsedMillis(analyzeStartNs)

            runOnUiThread {
                val uiStartNs = SystemClock.elapsedRealtimeNanos()
                latestCrossingSupportSnapshot = analysisResult.crossingSupportSnapshot
                renderOverlay(
                    bitmap = rotatedBitmap,
                    analysisResult = analysisResult,
                    showRawBoxes = rawDetectionSwitch.isChecked,
                    analysisOutputTransform = analysisOutputTransform
                )
                updateTargetInfo(analysisResult, enableTrafficLogic)
                updateDecisionDebugInfo(analysisResult, enableTrafficLogic)
                updateGpsDebugMapButtonState()
                updateUserStatus(analysisResult, enableTrafficLogic)
                updateStatusBorder(analysisResult.userGuidanceState, enableTrafficLogic)
                logDecisionIfChanged(analysisResult, enableTrafficLogic)
                logMapIfChanged(analysisResult.crossingSupportSnapshot)

                if (enableTrafficLogic) {
                    crossingSupportManager.setCrossingWindowActive(
                        analysisResult.guidancePhase == GuidancePhase.WALK_ALLOWED
                    )
                    feedbackManager.onGuidanceStateChanged(analysisResult.userGuidanceState)
                } else {
                    crossingSupportManager.setCrossingWindowActive(false)
                    feedbackManager.clearState()
                }

                stageDurationsMs["ui"] = elapsedMillis(uiStartNs)
                updateDebugInfo(
                    inferenceTime = result.inferenceTime,
                    totalLatencyMs = elapsedMillis(frameStartNs),
                    stageDurationsMs = stageDurationsMs
                )
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "Error processing camera frame", e)
        } finally {
            if (!imageClosed) {
                imageProxy.close()
            }
        }
    }

    private fun renderOverlay(
        bitmap: Bitmap,
        analysisResult: SignalAnalysisResult,
        showRawBoxes: Boolean,
        analysisOutputTransform: OutputTransform?
    ) {
        if (showRawBoxes && showBBoxOverlay) {
            val mappedBoxes =
                mapBoxesToPreviewView(
                    boxes = analysisResult.boxesToShow,
                    sourceWidth = bitmap.width.toFloat(),
                    sourceHeight = bitmap.height.toFloat(),
                    analysisOutputTransform = analysisOutputTransform
                )
            if (mappedBoxes != null) {
                overlay.setResults(mappedBoxes, inViewCoordinates = true)
            } else {
                overlay.setInputImageSize(bitmap.width, bitmap.height)
                overlay.setResults(analysisResult.boxesToShow)
            }
        } else {
            overlay.setResults(emptyList(), inViewCoordinates = true)
        }
    }

    private fun mapBoxesToPreviewView(
        boxes: List<OverlayView.BoundingBox>,
        sourceWidth: Float,
        sourceHeight: Float,
        analysisOutputTransform: OutputTransform?
    ): List<OverlayView.BoundingBox>? {
        if (analysisOutputTransform == null) {
            return null
        }

        val previewOutputTransform = viewFinder.outputTransform ?: return null

        return try {
            val coordinateTransform =
                CoordinateTransform(analysisOutputTransform, previewOutputTransform)
            boxes.map { box ->
                val mappedRect = RectF(
                    box.box.left * sourceWidth,
                    box.box.top * sourceHeight,
                    box.box.right * sourceWidth,
                    box.box.bottom * sourceHeight
                )
                coordinateTransform.mapRect(mappedRect)
                OverlayView.BoundingBox(
                    box = mappedRect,
                    clsName = box.clsName,
                    score = box.score,
                    debugRatio = box.debugRatio,
                    isTarget = box.isTarget
                )
            }
        } catch (e: IllegalArgumentException) {
            Log.w("MainActivity", "Preview transform unavailable for overlay mapping", e)
            null
        }
    }

    private fun updateTargetInfo(
        analysisResult: SignalAnalysisResult,
        enableTrafficLogic: Boolean
    ) {
        targetInfoText.text = if (enableTrafficLogic) {
            buildString {
                appendLine("Target: ${analysisResult.targetClassName}")
                appendLine("Score : ${String.format(Locale.US, "%.2f", analysisResult.targetScore)}")
                if (analysisResult.targetBox != null && analysisResult.targetBox.debugRatio >= 0f) {
                    appendLine("Ratio : ${String.format(Locale.US, "%.2f", analysisResult.targetBox.debugRatio)}")
                }
                append(
                    if (analysisResult.hasBlockingRisk) {
                        "Risk  : ${analysisResult.blockingRiskLabels.joinToString(", ")}"
                    } else {
                        "Risk  : none"
                    }
                )
            }
        } else {
            "Logic Disabled"
        }
    }

    private fun updateDecisionDebugInfo(
        analysisResult: SignalAnalysisResult,
        enableTrafficLogic: Boolean
    ) {
        decisionDebugText.text = if (enableTrafficLogic) {
            val context = analysisResult.crossingSupportSnapshot
            val map = context.mapProximitySnapshot
            buildString {
                appendLine("Decision: ${analysisResult.userGuidanceState}")
                appendLine("Phase   : ${analysisResult.guidancePhase}")
                appendLine("Reason  : ${analysisResult.guidanceBlockReason}")
                appendLine("Traffic : ${analysisResult.trafficState}")
                appendLine(
                    "Motion  : ${context.hasRecentGyroMotion} | GPS: ${context.hasRecentLocationMovement} | Down: ${context.isLookingDown} | Up: ${context.isLookingUp}"
                )
                appendLine(
                    "Tilt    : abs=${String.format(Locale.US, "%.0f", context.currentTiltDegrees)}° | signed=${String.format(Locale.US, "%.0f", context.currentSignedTiltDegrees)}° | Keep: ${context.supportsWalkContinuation}"
                )
                appendLine(
                    "Window  : ${context.isCrossingWindowActive} | Dist: ${String.format(Locale.US, "%.1f", context.crossingWindowDistanceMeters)}m | Elapsed: ${context.crossingWindowElapsedMs}ms"
                )
                appendLine(
                    "GPSFix  : lat=${context.currentLocationLatitude?.let { String.format(Locale.US, "%.6f", it) } ?: "n/a"} | lon=${context.currentLocationLongitude?.let { String.format(Locale.US, "%.6f", it) } ?: "n/a"} | acc=${context.currentLocationAccuracyMeters?.let { String.format(Locale.US, "%.1f", it) + "m" } ?: "n/a"}"
                )
                append(
                    "Map     : near=${map.isNearKnownFeature}, kind=${map.matchedKind?.wireName ?: "none"}, dist=${map.distanceMeters?.let { String.format(Locale.US, "%.1f", it) + "m" } ?: "n/a"}, id=${shortMapId(map.matchedFeatureId)}, ver=${map.datasetVersion ?: "none"}"
                )
            }
        } else {
            "Decision: DISABLED"
        }
    }

    private fun shortMapId(
        matchedFeatureId: String?
    ): String {
        if (matchedFeatureId.isNullOrBlank()) {
            return "none"
        }
        return if (matchedFeatureId.length <= 24) {
            matchedFeatureId
        } else {
            matchedFeatureId.take(24) + "…"
        }
    }

    private fun updateGpsDebugMapButtonState() {
        val currentSnapshot = currentCrossingSupportSnapshot()
        val hasLocationPermission = hasAnyLocationPermission()
        val hasMatch =
            currentSnapshot.mapProximitySnapshot.matchedLatitude != null &&
                currentSnapshot.mapProximitySnapshot.matchedLongitude != null
        openGpsDebugMapButton.isEnabled = hasLocationPermission
        openGpsDebugMapButton.text =
            if (hasLocationPermission) {
                if (hasMatch) {
                    "GPS + 매칭 지도 보기"
                } else {
                    "현재 GPS를 지도에서 보기"
                }
            } else {
                "위치 권한 필요"
            }
    }

    private fun openGpsDebugMap() {
        val currentSnapshot = currentCrossingSupportSnapshot()
        val fallbackLocation = bestAvailableLocation()
        val lat = currentSnapshot.currentLocationLatitude ?: fallbackLocation?.latitude
        val lon = currentSnapshot.currentLocationLongitude ?: fallbackLocation?.longitude
        val mapSnapshot = currentSnapshot.mapProximitySnapshot
        startActivity(
            DebugMapActivity.newIntent(
                activity = this,
                currentLat = lat ?: Double.NaN,
                currentLon = lon ?: Double.NaN,
                currentAccMeters =
                    currentSnapshot.currentLocationAccuracyMeters
                        ?: if (fallbackLocation?.hasAccuracy() == true) fallbackLocation.accuracy else null,
                matchedLat = mapSnapshot.matchedLatitude,
                matchedLon = mapSnapshot.matchedLongitude,
                matchedKind = mapSnapshot.matchedKind?.wireName,
                matchedId = mapSnapshot.matchedFeatureId,
                matchedDistMeters = mapSnapshot.distanceMeters,
                mapVersion = mapSnapshot.datasetVersion,
                isNearKnownFeature = mapSnapshot.isNearKnownFeature
            )
        )
    }

    private fun currentCrossingSupportSnapshot(): CrossingSupportSnapshot {
        return if (::crossingSupportManager.isInitialized) {
            crossingSupportManager.snapshot()
        } else {
            latestCrossingSupportSnapshot
        }
    }

    private fun clearMapCaches() {
        val deletedEntries = MapDebugCacheManager.clearAll(this)
        KineticGuestSessionManager.from(this).invalidateSession()
        Toast.makeText(
            this,
            "지도 캐시 초기화 완료 (${deletedEntries}개 삭제)",
            Toast.LENGTH_SHORT
        ).show()
    }

    @SuppressLint("MissingPermission")
    private fun bestAvailableLocation(): Location? {
        if (!hasAnyLocationPermission()) {
            return null
        }

        val candidates =
            buildList {
                runCatching {
                    if (locationManager?.isProviderEnabled(LocationManager.GPS_PROVIDER) == true) {
                        locationManager?.getLastKnownLocation(LocationManager.GPS_PROVIDER)?.let { add(it) }
                    }
                }
                runCatching {
                    if (locationManager?.isProviderEnabled(LocationManager.NETWORK_PROVIDER) == true) {
                        locationManager?.getLastKnownLocation(LocationManager.NETWORK_PROVIDER)?.let { add(it) }
                    }
                }
            }
        return candidates.minWithOrNull(
            compareBy<Location> { if (it.hasAccuracy()) it.accuracy else Float.MAX_VALUE }
                .thenByDescending { it.time }
        )
    }

    private fun hasAnyLocationPermission(): Boolean {
        val hasFine =
            ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) ==
                PackageManager.PERMISSION_GRANTED
        val hasCoarse =
            ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) ==
                PackageManager.PERMISSION_GRANTED
        return hasFine || hasCoarse
    }

    private fun updateUserStatus(
        analysisResult: SignalAnalysisResult,
        enableTrafficLogic: Boolean
    ) {
        if (!enableTrafficLogic) {
            statusTitleText.text = "분석 일시중지"
            statusTitleText.setTextColor(Color.WHITE)
            statusDetailText.text = "신호 인식이 일시중지되었습니다"
            return
        }

        when (analysisResult.trafficState) {
            TrafficLightState.RED -> {
                statusTitleText.text = "멈추세요"
                statusTitleText.setTextColor(Color.parseColor("#FF6B6B"))
                statusDetailText.text = "빨간불"
            }

            TrafficLightState.GREEN -> {
                when (analysisResult.userGuidanceState) {
                    UserGuidanceState.GO -> {
                        statusTitleText.text = "건너세요"
                        statusTitleText.setTextColor(Color.parseColor("#51CF66"))
                        statusDetailText.text = "초록 전환이 확인되었습니다"
                    }

                    UserGuidanceState.WAIT,
                    UserGuidanceState.STOP -> {
                        statusTitleText.text = "잠시 기다리세요"
                        statusTitleText.setTextColor(Color.WHITE)
                        statusDetailText.text = if (analysisResult.guidanceBlockReason == GuidanceBlockReason.BLOCKING_RISK) {
                            "차량 또는 자전거를 확인했습니다"
                        } else if (analysisResult.guidanceBlockReason == GuidanceBlockReason.NEED_RED_BASELINE) {
                            "다음 신호 전환을 기다리고 있습니다"
                        } else {
                            "처음 본 초록불은 안내하지 않습니다"
                        }
                    }
                }
            }

            TrafficLightState.UNKNOWN -> {
                if (analysisResult.userGuidanceState == UserGuidanceState.GO) {
                    statusTitleText.text = "건너세요"
                    statusTitleText.setTextColor(Color.parseColor("#51CF66"))
                    statusDetailText.text = if (analysisResult.crossingSupportSnapshot.isLookingDown) {
                        "휴대폰을 들어 신호등 쪽을 비춰주세요"
                    } else {
                        "초록 신호를 다시 찾는 중입니다"
                    }
                } else {
                    statusTitleText.text = "잠시 기다리세요"
                    statusTitleText.setTextColor(Color.WHITE)
                    statusDetailText.text = if (analysisResult.guidanceBlockReason == GuidanceBlockReason.BLOCKING_RISK) {
                        "주변 위험 요소를 확인 중입니다"
                    } else if (analysisResult.targetBox == null) {
                        "신호등을 화면 중앙에 맞춰주세요"
                    } else {
                        "신호 상태를 확인하고 있습니다"
                    }
                }
            }
        }
    }

    private fun logDecisionIfChanged(
        analysisResult: SignalAnalysisResult,
        enableTrafficLogic: Boolean
    ) {
        val summary = if (!enableTrafficLogic) {
            "logic_disabled"
        } else {
            "guidance=${analysisResult.userGuidanceState}," +
                "phase=${analysisResult.guidancePhase}," +
                "reason=${analysisResult.guidanceBlockReason}," +
                "traffic=${analysisResult.trafficState}," +
                "context=${analysisResult.crossingSupportSnapshot.toDebugSummary()}," +
                "risk=${analysisResult.blockingRiskLabels.joinToString("|").ifBlank { "none" }}"
        }

        if (summary != lastLoggedDecisionSummary) {
            lastLoggedDecisionSummary = summary
            Log.i("VIA_GUIDANCE", summary)
        }
    }

    private fun logMapIfChanged(
        crossingSupportSnapshot: CrossingSupportSnapshot
    ) {
        val summary = crossingSupportSnapshot.mapProximitySnapshot.toDebugSummary()
        if (summary != lastLoggedMapSummary) {
            lastLoggedMapSummary = summary
            Log.i("VIA_MAP", summary)
        }
    }

    private fun updateStatusBorder(
        guidanceState: UserGuidanceState,
        enableTrafficLogic: Boolean
    ) {
        if (!enableTrafficLogic) {
            statusBorder.setBackgroundResource(R.drawable.border_transparent)
            return
        }

        when (guidanceState) {
            UserGuidanceState.STOP -> statusBorder.setBackgroundResource(R.drawable.border_red)
            UserGuidanceState.GO -> statusBorder.setBackgroundResource(R.drawable.border_green)
            UserGuidanceState.WAIT -> statusBorder.setBackgroundResource(R.drawable.border_transparent)
        }
    }

    private fun resetPerformanceStats() {
        performanceTracker.clear()
        cameraRateTracker.clear()
        inputFpsText.text = "Camera FPS: 0"
        avgFpsText.text = "Processed Avg FPS: 0"
        avgLatencyText.text = "Avg Latency: 0ms"
        fpsText.text = "Processed FPS: 0"
        latencyText.text = "Detect: 0ms | Total: 0ms"
        stageBreakdownText.text = "Stages: n/a"
    }

    private fun publishBackendStatus(statusText: String) {
        backendStatusText.text = statusText
    }

    private fun applySystemBarInsets() {
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { _, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            updateCardMargins(buildInfoCard, top = systemBars.top + 20, bottom = 0)
            updateCardMargins(topControlCard, top = systemBars.top + 20, bottom = 0)
            updateCardMargins(debugContainer, top = 16, bottom = systemBars.bottom + 16)
            updateCardMargins(statusPanel, top = 0, bottom = systemBars.bottom + 20)
            insets
        }
    }

    private fun updateCardMargins(view: View, top: Int, bottom: Int) {
        val layoutParams = view.layoutParams as? ViewGroup.MarginLayoutParams ?: return
        layoutParams.topMargin = top
        layoutParams.bottomMargin = bottom
        view.layoutParams = layoutParams
    }

    private fun elapsedMillis(startTimeNs: Long): Long =
        (SystemClock.elapsedRealtimeNanos() - startTimeNs) / 1_000_000L

    override fun onDestroy() {
        super.onDestroy()
        hasStartedCamera = false
        viewFinder.removeCallbacks(cameraColdStartRecoveryRunnable)
        cameraRecoveryAttempts = 0
        lastCameraFrameAtElapsedMs = 0L
        pendingFrame.getAndSet(null)?.close()
        cameraExecutor?.shutdown()
        processingExecutor?.shutdown()
        cameraManager?.stopCamera()
        detector?.close()
        feedbackManager.release()
        crossingSupportManager.stop()
        guidanceStateStabilizer.reset()
        signalAnalyzer.reset()
        lastLoggedMapSummary = null
    }
}
