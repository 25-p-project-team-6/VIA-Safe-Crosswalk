package kr.co.gachon.pproject6.via

import android.Manifest
import android.annotation.SuppressLint
import android.content.res.ColorStateList
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.RectF
import android.graphics.SurfaceTexture
import android.location.Location
import android.location.LocationManager
import android.media.MediaPlayer
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.view.ViewGroup
import android.view.KeyEvent
import android.widget.LinearLayout
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
import kr.co.gachon.pproject6.via.context.CrosswalkGuidanceMessageBuilder
import kr.co.gachon.pproject6.via.feedback.SignalFeedbackManager
import kr.co.gachon.pproject6.via.camera.CameraManager
import kr.co.gachon.pproject6.via.context.CrossingSupportManager
import kr.co.gachon.pproject6.via.context.CrossingSupportSnapshot
import kr.co.gachon.pproject6.via.input.RemoteButtonAction
import kr.co.gachon.pproject6.via.input.RemoteButtonPressClassifier
import kr.co.gachon.pproject6.via.guide.UsageGuideActivity
import kr.co.gachon.pproject6.via.ml.AdvisoryAssessment
import kr.co.gachon.pproject6.via.ml.AdvisoryState
import kr.co.gachon.pproject6.via.ml.DetectionLabels
import kr.co.gachon.pproject6.via.ml.GuidanceBlockReason
import kr.co.gachon.pproject6.via.ml.GuidancePhase
import kr.co.gachon.pproject6.via.ml.SignalAdvisoryEvaluator
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
import kr.co.gachon.pproject6.via.ml.withAdvisoryAssessment
import kr.co.gachon.pproject6.via.ml.withGuidanceSnapshot
import kr.co.gachon.pproject6.via.map.KineticGuestSessionManager
import kr.co.gachon.pproject6.via.map.MapDebugCacheManager
import kr.co.gachon.pproject6.via.onboarding.AppPreferences
import kr.co.gachon.pproject6.via.onboarding.OnboardingActivity
import kr.co.gachon.pproject6.via.safety.EmergencyContactActivity
import kr.co.gachon.pproject6.via.settings.SettingsActivity
import kr.co.gachon.pproject6.via.ui.OverlayView
import kr.co.gachon.pproject6.via.util.ImageUtils
import kr.co.gachon.pproject6.via.util.PerformanceTracker
import kr.co.gachon.pproject6.via.util.RateTracker
import org.tensorflow.lite.gpu.CompatibilityList
import java.util.Locale
import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import java.util.concurrent.TimeUnit

class MainActivity : AppCompatActivity() {
    private companion object {
        private const val CAMERA_COLD_START_TIMEOUT_MS = 3_500L
        private const val MAX_CAMERA_COLD_START_RECOVERIES = 1
        private const val VIDEO_REPLAY_TARGET_FRAME_INTERVAL_MS = 33L
        private const val VIDEO_REPLAY_CAPTURE_TIMEOUT_MS = 250L
    }

    private data class StatusVisualState(
        val iconText: String,
        val iconBackgroundColor: Int,
        val iconTextColor: Int,
        val tintColor: Int,
        val borderResId: Int,
        val badgeText: String?
    )

    private lateinit var viewFinder: PreviewView
    private lateinit var videoReplayFrameView: TextureView
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
    private lateinit var videoReplayButton: MaterialButton
    private lateinit var openGpsDebugMapButton: MaterialButton
    private lateinit var clearMapCacheButton: MaterialButton
    private lateinit var targetInfoText: TextView
    private lateinit var decisionDebugText: TextView
    private lateinit var tuningDebugText: TextView
    private lateinit var statusTitleText: TextView
    private lateinit var statusDetailText: TextView
    private lateinit var statusIconText: TextView
    private lateinit var statusBadgeText: TextView
    private lateinit var confidenceSliderLabel: TextView
    private lateinit var trafficConfidenceLabel: TextView
    private lateinit var downTiltLabel: TextView
    private lateinit var upTiltLabel: TextView
    private lateinit var confidenceSlider: Slider
    private lateinit var trafficConfidenceSlider: Slider
    private lateinit var downTiltSlider: Slider
    private lateinit var upTiltSlider: Slider
    private lateinit var gpuSwitch: com.google.android.material.switchmaterial.SwitchMaterial
    private lateinit var zoomSwitch: androidx.appcompat.widget.SwitchCompat
    private lateinit var rawDetectionSwitch: androidx.appcompat.widget.SwitchCompat
    private lateinit var trafficLogicSwitch: androidx.appcompat.widget.SwitchCompat
    private lateinit var highlightTargetSwitch: androidx.appcompat.widget.SwitchCompat
    private lateinit var debugContainer: View
    private lateinit var settingsButton: android.widget.ImageButton
    private lateinit var usageGuideButton: android.widget.ImageButton
    private lateinit var debugToggleButton: android.widget.ImageButton
    private lateinit var buildInfoCard: View
    private lateinit var debugShortcutCard: View
    private lateinit var topControlCard: View
    private lateinit var statusPanel: View
    private lateinit var statusBorder: View
    private lateinit var statusTintOverlay: View
    private var lastLoggedDecisionSummary: String? = null
    private var lastLoggedMapSummary: String? = null
    private var suppressGpuToggleCallback = false
    private var suppressZoomToggleCallback = false
    private lateinit var preferences: AppPreferences
    private val locationManager by lazy {
        getSystemService(LocationManager::class.java)
    }
    private val remoteButtonClassifier = RemoteButtonPressClassifier()
    private var latestCrossingSupportSnapshot: CrossingSupportSnapshot = CrossingSupportSnapshot()

    private var cameraManager: CameraManager? = null
    private var hasStartedCamera = false
    private var cameraRecoveryAttempts = 0
    private val videoReplayRunning = AtomicBoolean(false)
    private var videoReplayUri: Uri? = null
    private var pendingReplayUriAfterDetectorInit: Uri? = null
    private var videoReplayPlayer: MediaPlayer? = null
    private var videoReplaySurface: Surface? = null
    private var zoomCheckedBeforeReplay: Boolean? = null
    private var inputRateLabel = "Camera FPS"
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
    
    private var currentModelName = "best_7cls_v2_float16_320.tflite" // Safe default for the 7-class pedestrian/vehicle signal model
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
    private val advisoryEvaluator = SignalAdvisoryEvaluator(GuidanceTuningDefaults.advisoryConfig)
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

    private val pickVideoReplayLauncher =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
            uri ?: return@registerForActivityResult
            runCatching {
                contentResolver.takePersistableUriPermission(
                    uri,
                    Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            }
            startVideoReplay(uri)
        }

    private val settingsLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            applyUserFeedbackPreferences()
            if (result.resultCode == RESULT_OK &&
                result.data?.getBooleanExtra(SettingsActivity.EXTRA_OPEN_DEBUG_PANEL, false) == true
            ) {
                setDebugPanelVisible(true)
            }
        }

    // 7-class fine-tuned model labels. Only human_* labels drive pedestrian signal guidance.
    private val finetunedLabels = DetectionLabels.sevenClassLabels

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
        videoReplayFrameView = findViewById(R.id.videoReplayFrameView)
        statusTintOverlay = findViewById(R.id.statusTintOverlay)
        overlay = findViewById(R.id.overlay)
        debugContainer = findViewById(R.id.debugContainer)
        settingsButton = findViewById(R.id.settingsButton)
        usageGuideButton = findViewById(R.id.usageGuideButton)
        debugToggleButton = findViewById(R.id.debugToggleButton)
        buildInfoCard = findViewById(R.id.buildInfoCard)
        debugShortcutCard = findViewById(R.id.debugShortcutCard)
        topControlCard = findViewById(R.id.topControlCard)
        statusPanel = findViewById(R.id.statusPanel)
        backendStatusText = findViewById(R.id.backendStatusText)
        resetAppButton = findViewById(R.id.resetAppButton)
        videoReplayButton = findViewById(R.id.videoReplayButton)
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
        addMainActionButtons()
        statusIconText = findViewById(R.id.statusIconText)
        statusBadgeText = findViewById(R.id.statusBadgeText)
        confidenceSliderLabel = findViewById(R.id.confidenceSliderLabel)
        trafficConfidenceLabel = findViewById(R.id.trafficConfidenceLabel)
        downTiltLabel = findViewById(R.id.downTiltLabel)
        upTiltLabel = findViewById(R.id.upTiltLabel)
        confidenceSlider = findViewById(R.id.confidenceSlider)
        trafficConfidenceSlider = findViewById(R.id.trafficConfidenceSlider)
        downTiltSlider = findViewById(R.id.downTiltSlider)
        upTiltSlider = findViewById(R.id.upTiltSlider)
        gpuSwitch = findViewById(R.id.gpuSwitch)
        zoomSwitch = findViewById(R.id.swZoom2x)
        rawDetectionSwitch = findViewById(R.id.swRawDetection)
        trafficLogicSwitch = findViewById(R.id.swTrafficLogic)
        highlightTargetSwitch = findViewById(R.id.swHighlightTarget)
        statusBorder = findViewById(R.id.statusBorder)
        feedbackManager = SignalFeedbackManager(this)
        applyUserFeedbackPreferences()
        crossingSupportManager = CrossingSupportManager(this, GuidanceTuningDefaults.crossingSupportConfig)
        buildInfoText.text = "v${BuildConfig.VERSION_NAME} (${BuildConfig.VERSION_CODE}) · ${BuildConfig.BUILD_STAMP}"
        updateTuningDebugText()
        Log.i("VIA_GUIDANCE", "tuning=${GuidanceTuningDefaults.toDebugSummary()}")
        modelNameText.text = "모델: $currentModelName"
        debugShortcutCard.visibility = if (BuildConfig.DEBUG) View.VISIBLE else View.GONE

        setDebugPanelVisible(showDebugInfo)

        applySystemBarInsets()

        settingsButton.setOnClickListener {
            settingsLauncher.launch(Intent(this, SettingsActivity::class.java))
        }
        usageGuideButton.setOnClickListener {
            startActivity(Intent(this, UsageGuideActivity::class.java))
        }
        debugToggleButton.setOnClickListener {
            setDebugPanelVisible(!showDebugInfo)
        }
        videoReplayButton.setOnClickListener {
            if (videoReplayRunning.get()) {
                stopVideoReplay(restoreCamera = true, clearSelectedVideo = true)
            } else {
                pickVideoReplayLauncher.launch(arrayOf("video/*"))
            }
        }
        resetAppButton.setOnClickListener {
            stopVideoReplay(restoreCamera = false, clearSelectedVideo = true)
            detector?.close()
            detector = null
            cameraManager?.stopCamera()
            preferences.clearAll()
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
        upTiltSlider.addOnChangeListener { _, value, _ ->
            crossingSupportManager.updateLookingUpThresholdDegrees(value)
            updateUpTiltLabel()
            updateTuningDebugText()
        }

        confidenceSlider.value = 0.5f
        trafficConfidenceSlider.value = 0.15f
        downTiltSlider.value = crossingSupportManager.currentLookingDownThresholdDegrees()
        upTiltSlider.value = crossingSupportManager.currentLookingUpThresholdDegrees()
        downTiltSlider.isEnabled = true
        upTiltSlider.isEnabled = true
        updateDownTiltLabel()
        updateUpTiltLabel()

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

    private fun addMainActionButtons() {
        val statusContent =
            (statusPanel as? ViewGroup)?.getChildAt(0) as? LinearLayout
                ?: error("statusPanel content must be a LinearLayout")
        val actionRow =
            LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                layoutParams =
                    LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.MATCH_PARENT,
                        LinearLayout.LayoutParams.WRAP_CONTENT
                    ).apply {
                        topMargin = dp(18)
                    }
            }
        actionRow.addView(
            mainActionButton(
                label = "횡단보도 안내",
                description = "가까운 횡단보도 거리와 방향 안내",
                endMargin = 6
            ) {
                announceNearbyCrosswalk()
            }
        )
        actionRow.addView(
            mainActionButton(
                label = "비상 연락",
                description = "비상 문자 5초 유예 화면 열기",
                startMargin = 6
            ) {
                openEmergencyContact(autoStartCountdown = true)
            }
        )
        statusContent.addView(actionRow)
    }

    private fun mainActionButton(
        label: String,
        description: String,
        startMargin: Int = 0,
        endMargin: Int = 0,
        onClick: () -> Unit
    ): MaterialButton = MaterialButton(this).apply {
        text = label
        contentDescription = description
        isAllCaps = false
        textSize = 15f
        minHeight = dp(56)
        cornerRadius = dp(18)
        layoutParams =
            LinearLayout.LayoutParams(
                0,
                LinearLayout.LayoutParams.WRAP_CONTENT,
                1f
            ).apply {
                marginStart = dp(startMargin)
                marginEnd = dp(endMargin)
            }
        setOnClickListener {
            onClick()
        }
    }

    private fun dp(value: Int): Int {
        return (value * resources.displayMetrics.density).toInt()
    }

    override fun dispatchKeyEvent(event: KeyEvent): Boolean {
        if (event.keyCode != KeyEvent.KEYCODE_SPACE) {
            return super.dispatchKeyEvent(event)
        }

        val action =
            when (event.action) {
                KeyEvent.ACTION_DOWN ->
                    remoteButtonClassifier.onDown(
                        eventTimeMs = event.eventTime,
                        repeatCount = event.repeatCount
                    )
                KeyEvent.ACTION_UP ->
                    remoteButtonClassifier.onUp(eventTimeMs = event.eventTime)
                else -> null
            }
        action?.let { handleRemoteButtonAction(it) }
        return true
    }

    private fun handleRemoteButtonAction(action: RemoteButtonAction) {
        when (action) {
            RemoteButtonAction.SHORT_PRESS -> {
                Log.i("VIA_REMOTE", "Space short press: nearby crosswalk guidance")
                Toast.makeText(this, "리모컨: 주변 횡단보도 안내", Toast.LENGTH_SHORT).show()
                announceNearbyCrosswalk()
            }
            RemoteButtonAction.LONG_PRESS -> {
                Log.i("VIA_REMOTE", "Space long press: emergency SMS countdown")
                Toast.makeText(this, "리모컨: 비상 문자 5초 유예", Toast.LENGTH_SHORT).show()
                openEmergencyContact(autoStartCountdown = true)
            }
        }
    }

    override fun onResume() {
        super.onResume()
        if (::feedbackManager.isInitialized) {
            applyUserFeedbackPreferences()
        }
        crossingSupportManager.start()
        latestCrossingSupportSnapshot = crossingSupportManager.snapshot()
        updateGpsDebugMapButtonState()
    }

    override fun onPause() {
        stopVideoReplay(restoreCamera = false, clearSelectedVideo = false)
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
        
        // 1. Human and vehicle traffic lights -> Traffic Slider.
        // Vehicle traffic lights are detected for uncertainty/debug only and never drive GO.
        specificMap[DetectionLabels.HUMAN_GREEN] = trafficLightThreshold
        specificMap[DetectionLabels.HUMAN_RED] = trafficLightThreshold
        specificMap[DetectionLabels.VEHICLE_GREEN] = trafficLightThreshold
        specificMap[DetectionLabels.VEHICLE_RED] = trafficLightThreshold
        
        // 2. Verified Objects -> General Slider
        val others = listOf(DetectionLabels.BICYCLE, DetectionLabels.MOTORCYCLE, DetectionLabels.VEHICLE)
        for (label in others) {
            specificMap[label] = generalObjThreshold
        }
        
        detector?.specificConfidenceThresholds = specificMap
    }

    private fun discoverModelFiles(): List<String> {
        return try {
            assets.list("")
                ?.let { DetectionLabels.modelFilesForActiveSchema(it.toList()) }
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
        val replayUriToResume = if (videoReplayRunning.get()) videoReplayUri else null
        if (replayUriToResume != null) {
            stopVideoReplay(restoreCamera = false, clearSelectedVideo = false)
            pendingReplayUriAfterDetectorInit = replayUriToResume
        }

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
            "Tuning: ${GuidanceTuningDefaults.toDebugSummary()}, tilt raw down=-160..-" +
                "${String.format(Locale.US, "%.0f", crossingSupportManager.currentLookingDownThresholdDegrees())}, " +
                "up=90..${String.format(Locale.US, "%.0f", crossingSupportManager.currentLookingUpThresholdDegrees())}"
    }

    private fun updateDownTiltLabel() {
        downTiltLabel.text =
            "Down Tilt Range: -160..-${String.format(Locale.US, "%.0f", crossingSupportManager.currentLookingDownThresholdDegrees())}"
    }

    private fun updateUpTiltLabel() {
        upTiltLabel.text =
            "Up Tilt Range: 90..${String.format(Locale.US, "%.0f", crossingSupportManager.currentLookingUpThresholdDegrees())}"
    }

    private fun setDebugPanelVisible(visible: Boolean) {
        showDebugInfo = visible
        debugContainer.visibility = if (visible) View.VISIBLE else View.GONE
        debugToggleButton.contentDescription =
            if (visible) "디버그 정보 닫기" else "디버그 정보 열기"
    }

    private fun applyUserFeedbackPreferences() {
        feedbackManager.voiceEnabled = preferences.voiceGuidanceEnabled
        feedbackManager.hapticEnabled = preferences.hapticFeedbackEnabled
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
        inputFpsText.text = cameraRateTracker.rateStr.replaceFirst("Camera FPS", inputRateLabel)
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
                    specificIouThresholds =
                        mapOf(
                            DetectionLabels.HUMAN_RED to 0.05f,
                            DetectionLabels.HUMAN_GREEN to 0.05f,
                            DetectionLabels.VEHICLE_RED to 0.05f,
                            DetectionLabels.VEHICLE_GREEN to 0.05f
                        )
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
                        "모델: $modelName (${newDetector.runtimeBackendLabel})"
                    Log.i("VIA_GPU", backendStatus)
                    publishBackendStatus("Backend: ${newDetector.runtimeBackendLabel}")
                    val pendingReplayUri = pendingReplayUriAfterDetectorInit
                    if (pendingReplayUri != null) {
                        pendingReplayUriAfterDetectorInit = null
                        startVideoReplay(pendingReplayUri)
                    } else if (restartCameraAfterInit && hasStartedCamera) {
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
        videoReplayRunning.set(false)
        videoReplayFrameView.visibility = View.GONE
        viewFinder.visibility = View.VISIBLE
        inputRateLabel = "Camera FPS"
        videoReplayButton.text = "샘플 영상 선택"
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
        if (videoReplayRunning.get()) {
            videoReplayFrameView.pivotX = videoReplayFrameView.width / 2f
            videoReplayFrameView.pivotY = videoReplayFrameView.height / 2f
            videoReplayFrameView.scaleX = zoomRatio
            videoReplayFrameView.scaleY = zoomRatio
            return
        }
        cameraManager?.setZoom(zoomRatio)
    }

    private fun startVideoReplay(uri: Uri) {
        val executor = processingExecutor ?: return
        videoReplayUri = uri
        videoReplayRunning.set(true)
        inputRateLabel = "Replay FPS"
        resetPerformanceStats()
        guidanceRuntimeResetter.resetForTrafficLogicDisabled()
        viewFinder.removeCallbacks(cameraColdStartRecoveryRunnable)
        cameraManager?.stopCamera()
        hasStartedCamera = false
        pendingFrame.getAndSet(null)?.close()

        viewFinder.visibility = View.GONE
        videoReplayFrameView.visibility = View.VISIBLE
        videoReplayButton.text = "샘플 영상 중지"
        publishBackendStatus("Replay: preparing video input")
        zoomCheckedBeforeReplay = zoomSwitch.isChecked
        suppressZoomToggleCallback = true
        zoomSwitch.isChecked = false
        suppressZoomToggleCallback = false
        zoomSwitch.isEnabled = true
        zoomSwitch.text = "Use 2x Zoom (Replay, optional)"
        applySelectedZoom()
        prepareVideoReplayPlayer(uri)

        executor.execute {
            try {
                while (videoReplayRunning.get()) {
                    if (!videoReplayFrameView.isAvailable || detector == null) {
                        Thread.sleep(50L)
                        continue
                    }

                    val frameStartNs = SystemClock.elapsedRealtimeNanos()
                    val captureStartNs = SystemClock.elapsedRealtimeNanos()
                    val resolution = currentModelProfile.analysisResolution
                    val bitmap = captureReplayBitmap(
                        width = resolution.width,
                        height = resolution.height
                    )

                    if (bitmap == null) {
                        Thread.sleep(50L)
                        continue
                    }

                    val stageDurationsMs =
                        linkedMapOf("capture" to elapsedMillis(captureStartNs))

                    cameraRateTracker.mark()
                    processBitmapFrame(
                        bitmap = bitmap,
                        frameStartNs = frameStartNs,
                        stageDurationsMs = stageDurationsMs,
                        analysisOutputTransform = null,
                        useLiveContext = false,
                        mapOverlayDirectlyToView = true
                    )

                    val elapsedMs = elapsedMillis(frameStartNs)
                    val remainingFrameBudgetMs =
                        VIDEO_REPLAY_TARGET_FRAME_INTERVAL_MS - elapsedMs
                    if (remainingFrameBudgetMs > 0L) {
                        Thread.sleep(remainingFrameBudgetMs)
                    }
                }
            } catch (e: Exception) {
                Log.e("MainActivity", "Error replaying sample video", e)
                runOnUiThread {
                    Toast.makeText(
                        this,
                        "샘플 영상 재생 실패: ${e.message}",
                        Toast.LENGTH_LONG
                    ).show()
                    stopVideoReplay(restoreCamera = true, clearSelectedVideo = true)
                }
            }
        }
    }

    private fun prepareVideoReplayPlayer(uri: Uri) {
        releaseVideoReplayPlayer()

        fun attach(surfaceTexture: SurfaceTexture) {
            val surface = Surface(surfaceTexture)
            videoReplaySurface = surface
            val player = MediaPlayer()
            videoReplayPlayer = player
            try {
                player.setDataSource(this, uri)
                player.setSurface(surface)
                player.isLooping = true
                player.setOnPreparedListener {
                    if (videoReplayRunning.get()) {
                        it.start()
                        publishBackendStatus("Replay: video input")
                    }
                }
                player.setOnErrorListener { _, what, extra ->
                    Log.w("VIA_REPLAY", "MediaPlayer playback error what=$what extra=$extra")
                    Toast.makeText(
                        this,
                        "샘플 영상 재생 실패: $what/$extra",
                        Toast.LENGTH_LONG
                    ).show()
                    stopVideoReplay(restoreCamera = true, clearSelectedVideo = true)
                    true
                }
                player.prepareAsync()
            } catch (e: Exception) {
                Log.e("VIA_REPLAY", "Failed to prepare sample video", e)
                Toast.makeText(this, "샘플 영상 준비 실패: ${e.message}", Toast.LENGTH_LONG).show()
                stopVideoReplay(restoreCamera = true, clearSelectedVideo = true)
            }
        }

        if (videoReplayFrameView.isAvailable) {
            videoReplayFrameView.surfaceTexture?.let(::attach)
        } else {
            videoReplayFrameView.surfaceTextureListener =
                object : TextureView.SurfaceTextureListener {
                    override fun onSurfaceTextureAvailable(
                        surface: SurfaceTexture,
                        width: Int,
                        height: Int
                    ) {
                        attach(surface)
                    }

                    override fun onSurfaceTextureSizeChanged(
                        surface: SurfaceTexture,
                        width: Int,
                        height: Int
                    ) = Unit

                    override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
                        videoReplayRunning.set(false)
                        releaseVideoReplayPlayer()
                        return true
                    }

                    override fun onSurfaceTextureUpdated(surface: SurfaceTexture) = Unit
                }
        }
    }

    private fun captureReplayBitmap(width: Int, height: Int): Bitmap? {
        if (!videoReplayRunning.get()) {
            return null
        }

        val latch = CountDownLatch(1)
        val bitmapRef = AtomicReference<Bitmap?>()
        runOnUiThread {
            try {
                if (videoReplayRunning.get() && videoReplayFrameView.isAvailable) {
                    val replayBitmap = videoReplayFrameView.getBitmap(width, height)
                    bitmapRef.set(applyReplayZoomCropIfNeeded(replayBitmap, width, height))
                }
            } finally {
                latch.countDown()
            }
        }
        latch.await(VIDEO_REPLAY_CAPTURE_TIMEOUT_MS, TimeUnit.MILLISECONDS)
        return bitmapRef.get()
    }

    private fun applyReplayZoomCropIfNeeded(
        bitmap: Bitmap?,
        outputWidth: Int,
        outputHeight: Int
    ): Bitmap? {
        if (bitmap == null || !zoomSwitch.isChecked || !zoomSwitch.isEnabled) {
            return bitmap
        }
        val cropWidth = (bitmap.width / 2).coerceAtLeast(1)
        val cropHeight = (bitmap.height / 2).coerceAtLeast(1)
        val left = ((bitmap.width - cropWidth) / 2).coerceAtLeast(0)
        val top = ((bitmap.height - cropHeight) / 2).coerceAtLeast(0)
        val cropped = Bitmap.createBitmap(bitmap, left, top, cropWidth, cropHeight)
        return Bitmap.createScaledBitmap(cropped, outputWidth, outputHeight, true)
    }

    private fun releaseVideoReplayPlayer() {
        videoReplayPlayer?.let { player ->
            runCatching {
                if (player.isPlaying) {
                    player.stop()
                }
            }
            runCatching { player.release() }
        }
        videoReplayPlayer = null
        videoReplaySurface?.release()
        videoReplaySurface = null
    }

    private fun stopVideoReplay(
        restoreCamera: Boolean,
        clearSelectedVideo: Boolean
    ) {
        videoReplayRunning.set(false)
        if (clearSelectedVideo) {
            videoReplayUri = null
        }
        pendingReplayUriAfterDetectorInit = null
        if (!::videoReplayFrameView.isInitialized) {
            return
        }
        releaseVideoReplayPlayer()
        videoReplayFrameView.surfaceTextureListener = null
        videoReplayFrameView.scaleX = 1f
        videoReplayFrameView.scaleY = 1f
        videoReplayFrameView.visibility = View.GONE
        viewFinder.visibility = View.VISIBLE
        videoReplayButton.text = "샘플 영상 선택"
        zoomCheckedBeforeReplay?.let { previousZoom ->
            suppressZoomToggleCallback = true
            zoomSwitch.isChecked = previousZoom
            suppressZoomToggleCallback = false
        }
        zoomCheckedBeforeReplay = null
        overlay.clear()
        inputRateLabel = "Camera FPS"
        resetPerformanceStats()
        guidanceRuntimeResetter.resetForTrafficLogicDisabled()
        publishBackendStatus("Backend: ${detector?.runtimeBackendLabel ?: "unknown"}")

        if (restoreCamera &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        }
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

            processBitmapFrame(
                bitmap = rotatedBitmap,
                frameStartNs = frameStartNs,
                stageDurationsMs = stageDurationsMs,
                analysisOutputTransform = analysisOutputTransform,
                useLiveContext = true,
                mapOverlayDirectlyToView = false
            )
        } catch (e: Exception) {
            Log.e("MainActivity", "Error processing camera frame", e)
        } finally {
            if (!imageClosed) {
                imageProxy.close()
            }
        }
    }

    private fun processBitmapFrame(
        bitmap: Bitmap,
        frameStartNs: Long,
        stageDurationsMs: LinkedHashMap<String, Long>,
        analysisOutputTransform: OutputTransform?,
        useLiveContext: Boolean,
        mapOverlayDirectlyToView: Boolean
    ) {
        val activeDetector = detector ?: return

        val detectStartNs = SystemClock.elapsedRealtimeNanos()
        val result = activeDetector.detect(bitmap, confidenceThreshold)
        stageDurationsMs["detect"] = elapsedMillis(detectStartNs)

        val enableTrafficLogic = trafficLogicSwitch.isChecked
        val crossingSupportSnapshot =
            if (enableTrafficLogic && useLiveContext) {
                crossingSupportManager.snapshot()
            } else {
                CrossingSupportSnapshot()
            }
        val analyzeStartNs = SystemClock.elapsedRealtimeNanos()
        val rawAnalysisResult = signalAnalyzer.analyze(
            bitmap = bitmap,
            rawBoxes = result.boxes,
            enableTrafficLogic = enableTrafficLogic,
            enableHighlight = highlightTargetSwitch.isChecked,
            crossingSupportSnapshot = crossingSupportSnapshot
        )
        val stabilizedAnalysisResult =
            if (enableTrafficLogic) {
                rawAnalysisResult.withGuidanceSnapshot(
                    guidanceStateStabilizer.stabilize(rawAnalysisResult.toGuidanceSnapshot())
                )
            } else {
                guidanceStateStabilizer.reset()
                rawAnalysisResult
            }
        val analysisResult =
            if (enableTrafficLogic) {
                stabilizedAnalysisResult.withAdvisoryAssessment(
                    advisoryEvaluator.evaluate(stabilizedAnalysisResult)
                )
            } else {
                stabilizedAnalysisResult
            }
        stageDurationsMs["analyze"] = elapsedMillis(analyzeStartNs)

        runOnUiThread {
            val uiStartNs = SystemClock.elapsedRealtimeNanos()
            latestCrossingSupportSnapshot = analysisResult.crossingSupportSnapshot
            renderOverlay(
                bitmap = bitmap,
                analysisResult = analysisResult,
                showRawBoxes = rawDetectionSwitch.isChecked,
                analysisOutputTransform = analysisOutputTransform,
                mapOverlayDirectlyToView = mapOverlayDirectlyToView
            )
            updateTargetInfo(analysisResult, enableTrafficLogic)
            updateDecisionDebugInfo(analysisResult, enableTrafficLogic)
            updateGpsDebugMapButtonState()
            updateUserStatus(analysisResult, enableTrafficLogic)
            updateStatusVisuals(analysisResult, enableTrafficLogic)
            logDecisionIfChanged(analysisResult, enableTrafficLogic)
            logMapIfChanged(analysisResult.crossingSupportSnapshot)

            if (enableTrafficLogic) {
                crossingSupportManager.setCrossingWindowActive(
                    useLiveContext && analysisResult.guidancePhase == GuidancePhase.WALK_ALLOWED
                )
                feedbackManager.onAdvisoryChanged(
                    AdvisoryAssessment(
                        state = analysisResult.advisoryState,
                        confidenceLevel = analysisResult.advisoryConfidenceLevel,
                        confidenceScore = analysisResult.advisoryConfidenceScore,
                        confidenceReasons = analysisResult.advisoryConfidenceReasons,
                        titleText = analysisResult.advisoryTitleText,
                        detailText = analysisResult.advisoryDetailText,
                        speechText = analysisResult.advisorySpeechText
                    )
                )
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
    }

    private fun renderOverlay(
        bitmap: Bitmap,
        analysisResult: SignalAnalysisResult,
        showRawBoxes: Boolean,
        analysisOutputTransform: OutputTransform?,
        mapOverlayDirectlyToView: Boolean
    ) {
        if (showRawBoxes && showBBoxOverlay) {
            val mappedBoxes =
                if (mapOverlayDirectlyToView) {
                    mapNormalizedBoxesDirectlyToOverlay(analysisResult.boxesToShow)
                } else {
                    mapBoxesToPreviewView(
                        boxes = analysisResult.boxesToShow,
                        sourceWidth = bitmap.width.toFloat(),
                        sourceHeight = bitmap.height.toFloat(),
                        analysisOutputTransform = analysisOutputTransform
                    )
                }
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

    private fun mapNormalizedBoxesDirectlyToOverlay(
        boxes: List<OverlayView.BoundingBox>
    ): List<OverlayView.BoundingBox> {
        val viewWidth = overlay.width.toFloat().takeIf { it > 0f } ?: videoReplayFrameView.width.toFloat()
        val viewHeight = overlay.height.toFloat().takeIf { it > 0f } ?: videoReplayFrameView.height.toFloat()
        return boxes.map { box ->
            OverlayView.BoundingBox(
                box = RectF(
                    box.box.left * viewWidth,
                    box.box.top * viewHeight,
                    box.box.right * viewWidth,
                    box.box.bottom * viewHeight
                ),
                clsName = box.clsName,
                score = box.score,
                debugRatio = box.debugRatio,
                isTarget = box.isTarget
            )
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
                    if (analysisResult.occupancyCaution) {
                        "Caution: ${analysisResult.occupancyCautionLabels.joinToString(", ")}"
                    } else {
                        "Caution: none"
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
                appendLine(
                    "Context : tier=${analysisResult.guidanceContinuityTier} | handoff=${analysisResult.handoffDecision} | caution=${analysisResult.occupancyCaution}"
                )
                appendLine(
                    "Advisory: ${analysisResult.advisoryState} | conf=${analysisResult.advisoryConfidenceLevel}(${analysisResult.advisoryConfidenceScore}) | humanSignals=${analysisResult.trafficLightCount} | vehicleSignals=${analysisResult.vehicleTrafficLightCount} | zoom=${analysisResult.needsZoomSuggestion} | reacquire=${analysisResult.targetRecentlyReacquired} | clusterChanges=${analysisResult.recentMatchedClusterChangeCount}"
                )
                append(
                    "Map     : near=${map.isNearKnownFeature}, kind=${map.matchedKind?.wireName ?: "none"}, source=${map.matchedSource?.wireName ?: "none"}, dist=${map.distanceMeters?.let { String.format(Locale.US, "%.1f", it) + "m" } ?: "n/a"}, cluster=${shortMapId(map.matchedClusterId)}, members=${map.matchedMemberCount}, transition=${map.clusterTransitionKind.wireName}, ver=${map.datasetVersion ?: "none"}"
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
                matchedSource = mapSnapshot.matchedSource?.wireName,
                matchedId = mapSnapshot.matchedFeatureId,
                matchedDistMeters = mapSnapshot.distanceMeters,
                mapVersion = mapSnapshot.datasetVersion,
                isNearKnownFeature = mapSnapshot.isNearKnownFeature
            )
        )
    }

    private fun announceNearbyCrosswalk() {
        val guidanceMessage =
            if (!hasAnyLocationPermission()) {
                CrosswalkGuidanceMessageBuilder.build(CrossingSupportSnapshot())
            } else {
                CrosswalkGuidanceMessageBuilder.build(crosswalkGuidanceSnapshot())
            }
        Toast.makeText(this, guidanceMessage.detail, Toast.LENGTH_LONG).show()
        feedbackManager.speakImmediate(
            message = guidanceMessage.speechText,
            utteranceId = "nearby_crosswalk_guidance"
        )
    }

    private fun openEmergencyContact(autoStartCountdown: Boolean) {
        val intent = Intent(this, EmergencyContactActivity::class.java).apply {
            putExtra(EmergencyContactActivity.EXTRA_AUTO_START_COUNTDOWN, autoStartCountdown)
        }
        startActivity(intent)
    }

    private fun crosswalkGuidanceSnapshot(): CrossingSupportSnapshot {
        val currentSnapshot = currentCrossingSupportSnapshot()
        if (currentSnapshot.currentLocationLatitude != null &&
            currentSnapshot.currentLocationLongitude != null
        ) {
            return currentSnapshot
        }

        val fallbackLocation = bestAvailableLocation() ?: return currentSnapshot
        return currentSnapshot.copy(
            currentLocationLatitude = fallbackLocation.latitude,
            currentLocationLongitude = fallbackLocation.longitude,
            currentLocationAccuracyMeters =
                if (fallbackLocation.hasAccuracy()) fallbackLocation.accuracy else null,
            currentHeadingDegrees =
                currentSnapshot.currentHeadingDegrees
                    ?: if (fallbackLocation.hasBearing()) fallbackLocation.bearing else null
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
            statusPanel.contentDescription = "현재 상태: 분석 일시중지. 신호 인식이 일시중지되었습니다."
            return
        }
        statusTitleText.text = analysisResult.advisoryTitleText
        statusTitleText.setTextColor(
            when (analysisResult.advisoryState) {
                AdvisoryState.RED_CONFIRMED -> Color.parseColor("#FF6B6B")
                AdvisoryState.GREEN_CONFIRMED,
                AdvisoryState.GREEN_WITH_CAUTION -> Color.parseColor("#51CF66")
                AdvisoryState.TRANSITION_WAIT,
                AdvisoryState.UNCERTAIN_VIEW -> Color.WHITE
            }
        )
        statusDetailText.text = analysisResult.advisoryDetailText
        statusPanel.contentDescription =
            "현재 상태: ${analysisResult.advisoryTitleText}. ${analysisResult.advisoryDetailText}"
    }

    private fun updateStatusVisuals(
        analysisResult: SignalAnalysisResult,
        enableTrafficLogic: Boolean
    ) {
        if (!enableTrafficLogic) {
            statusBorder.setBackgroundResource(R.drawable.border_transparent)
            statusTintOverlay.setBackgroundColor(Color.TRANSPARENT)
            statusTintOverlay.alpha = 0f
            statusIconText.text = "?"
            statusIconText.contentDescription = "분석 일시중지"
            statusIconText.setTextColor(ContextCompat.getColor(this, R.color.via_on_surface))
            statusIconText.backgroundTintList =
                ColorStateList.valueOf(ContextCompat.getColor(this, R.color.via_status_wait))
            statusBadgeText.visibility = View.INVISIBLE
            return
        }

        val visualState = deriveStatusVisualState(analysisResult)
        statusBorder.setBackgroundResource(visualState.borderResId)
        val tintColor =
            if (preferences.screenColorFeedbackEnabled) visualState.tintColor else Color.TRANSPARENT
        statusTintOverlay.setBackgroundColor(tintColor)
        statusTintOverlay.alpha = if (tintColor == Color.TRANSPARENT) 0f else 1f
        statusIconText.text = visualState.iconText
        statusIconText.contentDescription = statusTitleText.text
        statusIconText.setTextColor(visualState.iconTextColor)
        statusIconText.backgroundTintList = ColorStateList.valueOf(visualState.iconBackgroundColor)
        statusBadgeText.visibility = if (visualState.badgeText != null) View.VISIBLE else View.INVISIBLE
        statusBadgeText.text = visualState.badgeText ?: ""
        statusBadgeText.contentDescription = visualState.badgeText ?: ""
    }

    private fun deriveStatusVisualState(
        analysisResult: SignalAnalysisResult
    ): StatusVisualState {
        return when {
            analysisResult.advisoryState == AdvisoryState.RED_CONFIRMED ->
                StatusVisualState(
                    iconText = "■",
                    iconBackgroundColor = ContextCompat.getColor(this, R.color.via_status_stop),
                    iconTextColor = ContextCompat.getColor(this, R.color.via_on_primary),
                    tintColor = ContextCompat.getColor(this, R.color.via_status_stop_tint),
                    borderResId = R.drawable.border_red,
                    badgeText = null
                )

            analysisResult.advisoryState == AdvisoryState.GREEN_WITH_CAUTION ->
                StatusVisualState(
                    iconText = "!",
                    iconBackgroundColor = ContextCompat.getColor(this, R.color.via_status_caution),
                    iconTextColor = ContextCompat.getColor(this, R.color.via_status_badge_on),
                    tintColor = Color.parseColor("#3DFFC857"),
                    borderResId = R.drawable.border_green,
                    badgeText = "차량 주의"
                )

            analysisResult.advisoryState == AdvisoryState.GREEN_CONFIRMED ->
                StatusVisualState(
                    iconText = "▶",
                    iconBackgroundColor = ContextCompat.getColor(this, R.color.via_status_go),
                    iconTextColor = ContextCompat.getColor(this, R.color.via_on_primary),
                    tintColor = ContextCompat.getColor(this, R.color.via_status_go_tint),
                    borderResId = R.drawable.border_green,
                    badgeText = null
                )

            else ->
                StatusVisualState(
                    iconText = if (analysisResult.advisoryState == AdvisoryState.TRANSITION_WAIT) "⌛" else "?",
                    iconBackgroundColor = ContextCompat.getColor(this, R.color.via_status_wait),
                    iconTextColor = ContextCompat.getColor(this, R.color.via_on_primary),
                    tintColor = Color.TRANSPARENT,
                    borderResId = R.drawable.border_transparent,
                    badgeText = null
                )
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
                "advisory=${analysisResult.advisoryState}," +
                "confidence=${analysisResult.advisoryConfidenceLevel}:${analysisResult.advisoryConfidenceScore}," +
                "advisoryReasons=${analysisResult.advisoryConfidenceReasons.joinToString("|").ifBlank { "none" }}," +
                "signals=human:${analysisResult.trafficLightCount}|vehicle:${analysisResult.vehicleTrafficLightCount}," +
                "ambiguity=multi:${analysisResult.multipleSignalDetected}|zoom:${analysisResult.needsZoomSuggestion}|reacquire:${analysisResult.targetRecentlyReacquired}|clusterChanges:${analysisResult.recentMatchedClusterChangeCount}," +
                "tier=${analysisResult.guidanceContinuityTier}," +
                "handoff=${analysisResult.handoffDecision}," +
                "caution=${analysisResult.occupancyCaution}," +
                "traffic=${analysisResult.trafficState}," +
                "context=${analysisResult.crossingSupportSnapshot.toDebugSummary()}," +
                "occupancy=${analysisResult.occupancyCautionLabels.joinToString("|").ifBlank { "none" }}"
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

    private fun resetPerformanceStats() {
        performanceTracker.clear()
        cameraRateTracker.clear()
        inputFpsText.text = "$inputRateLabel: 0"
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
            updateCardMargins(debugShortcutCard, top = 8, bottom = 0)
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
        videoReplayRunning.set(false)
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
