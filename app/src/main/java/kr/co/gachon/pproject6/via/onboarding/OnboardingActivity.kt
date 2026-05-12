package kr.co.gachon.pproject6.via.onboarding

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.ImageProxy
import androidx.camera.view.PreviewView
import androidx.constraintlayout.widget.ConstraintLayout
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.google.android.material.button.MaterialButton
import kr.co.gachon.pproject6.via.MainActivity
import kr.co.gachon.pproject6.via.R
import kr.co.gachon.pproject6.via.BuildConfig
import kr.co.gachon.pproject6.via.camera.CameraManager
import kr.co.gachon.pproject6.via.ml.DetectionLabels
import kr.co.gachon.pproject6.via.ml.InferenceModelProfile
import kr.co.gachon.pproject6.via.ml.YoloDetector
import kr.co.gachon.pproject6.via.map.KineticGuestSessionManager
import kr.co.gachon.pproject6.via.safety.EmergencyContactActivity
import kr.co.gachon.pproject6.via.util.ImageUtils
import org.tensorflow.lite.gpu.CompatibilityList
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class OnboardingActivity : AppCompatActivity(), TextToSpeech.OnInitListener {
    private enum class Step {
        INTRO,
        PERMISSION,
        EMERGENCY_CONTACT,
        CALIBRATING,
        RESULT
    }

    private lateinit var titleText: TextView
    private lateinit var stepLabelText: TextView
    private lateinit var bodyText: TextView
    private lateinit var detailText: TextView
    private lateinit var actionButton: MaterialButton
    private lateinit var secondaryButton: MaterialButton
    private lateinit var replayButton: MaterialButton
    private lateinit var progressBar: ProgressBar
    private lateinit var progressText: TextView
    private lateinit var calibrationPreview: PreviewView
    private lateinit var rootView: View
    private lateinit var buildInfoCard: View
    private lateinit var buildInfoText: TextView
    private lateinit var headerCard: View
    private lateinit var previewCard: View
    private lateinit var actionContainer: View
    private lateinit var preferences: AppPreferences

    private val deviceSummary by lazy {
        "${Build.MANUFACTURER} ${Build.MODEL} / Android ${Build.VERSION.RELEASE}"
    }

    private var currentStep = Step.INTRO
    private var ttsReady = false
    private lateinit var tts: TextToSpeech
    private var spokenMessage: String? = null

    private var cameraExecutor: ExecutorService? = null
    private var cameraManager: CameraManager? = null
    @Volatile
    private var calibrationDetector: YoloDetector? = null
    private var reusableRotatedBitmap: Bitmap? = null

    private var candidateProfiles: List<InferenceModelProfile> = emptyList()
    private val calibrationResults = mutableListOf<CalibrationProfileResult>()
    private var currentCandidateIndex = -1
    private var activeRun: ActiveCalibrationRun? = null
    private var calibrationFinalized = false

    private val permissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { result ->
            val cameraGranted = result[Manifest.permission.CAMERA] == true || hasCameraPermission()
            val locationGranted =
                result[Manifest.permission.ACCESS_FINE_LOCATION] == true ||
                    result[Manifest.permission.ACCESS_COARSE_LOCATION] == true ||
                    hasLocationPermission()
            val smsGranted = result[Manifest.permission.SEND_SMS] == true || hasSmsPermission()
            if (OnboardingPermissionPolicy.hasRequiredPermissions(
                    hasCameraPermission = cameraGranted,
                    hasLocationPermission = locationGranted,
                    hasSmsPermission = smsGranted
                )
            ) {
                showEmergencyContactStep()
            } else {
                showPermissionStep(
                    detailOverride = "카메라, 위치, SMS 권한이 필요합니다. 다시 허용해 주세요."
                )
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        preferences = AppPreferences(this)
        if (preferences.onboardingCompleted) {
            launchMain()
            return
        }

        setContentView(R.layout.activity_onboarding)
        rootView = findViewById(R.id.onboardingRoot)
        buildInfoCard = findViewById(R.id.onboardingBuildInfoCard)
        buildInfoText = findViewById(R.id.onboardingBuildInfoText)
        headerCard = findViewById(R.id.onboardingHeaderCard)
        previewCard = findViewById(R.id.onboardingPreviewCard)
        actionContainer = findViewById(R.id.onboardingActionContainer)
        stepLabelText = findViewById(R.id.onboardingStepLabel)
        titleText = findViewById(R.id.onboardingTitleText)
        bodyText = findViewById(R.id.onboardingBodyText)
        detailText = findViewById(R.id.onboardingDetailText)
        actionButton = findViewById(R.id.onboardingActionButton)
        secondaryButton = findViewById(R.id.onboardingSecondaryButton)
        replayButton = findViewById(R.id.onboardingReplayButton)
        progressBar = findViewById(R.id.onboardingProgressBar)
        progressText = findViewById(R.id.onboardingProgressText)
        calibrationPreview = findViewById(R.id.onboardingPreview)
        buildInfoText.text = "v${BuildConfig.VERSION_NAME} (${BuildConfig.VERSION_CODE}) · ${BuildConfig.BUILD_STAMP}"
        cameraExecutor = Executors.newSingleThreadExecutor()
        tts = TextToSpeech(this, this)

        actionButton.setOnClickListener { handlePrimaryAction() }
        secondaryButton.setOnClickListener { handleSecondaryAction() }
        replayButton.setOnClickListener { speakCurrentStep(forceReplay = true) }
        applySystemBarInsets()

        showIntroStep()
    }

    override fun onResume() {
        super.onResume()
        if (currentStep == Step.EMERGENCY_CONTACT) {
            updateEmergencyContactStep()
        }
    }

    override fun onInit(status: Int) {
        if (status != TextToSpeech.SUCCESS) return
        val preferred = tts.setLanguage(Locale.KOREAN)
        if (preferred == TextToSpeech.LANG_MISSING_DATA || preferred == TextToSpeech.LANG_NOT_SUPPORTED) {
            tts.setLanguage(Locale.getDefault())
        }
        ttsReady = true
        speakCurrentStep(forceReplay = true)
    }

    private fun handlePrimaryAction() {
        when (currentStep) {
            Step.INTRO -> showPermissionStep()
            Step.PERMISSION -> {
                if (OnboardingPermissionPolicy.hasRequiredPermissions(
                        hasCameraPermission = hasCameraPermission(),
                        hasLocationPermission = hasLocationPermission(),
                        hasSmsPermission = hasSmsPermission()
                    )
                ) {
                    showEmergencyContactStep()
                } else {
                    requestOnboardingPermissions()
                }
            }
            Step.EMERGENCY_CONTACT -> {
                if (hasEmergencyContact()) {
                    startCalibration()
                } else {
                    openEmergencyContactSetup(markOnboardingComplete = false)
                }
            }
            Step.CALIBRATING -> Unit
            Step.RESULT -> {
                preferences.onboardingCompleted = true
                launchMain()
            }
        }
    }

    private fun handleSecondaryAction() {
        when (currentStep) {
            Step.EMERGENCY_CONTACT -> startCalibration()
            Step.RESULT -> {
                openEmergencyContactSetup(markOnboardingComplete = true)
            }
            else -> speakCurrentStep(forceReplay = true)
        }
    }

    private fun showIntroStep() {
        currentStep = Step.INTRO
        applyLayoutMode(infoMode = true)
        calibrationPreview.visibility = View.GONE
        previewCard.visibility = View.GONE
        progressBar.visibility = View.GONE
        progressText.visibility = View.GONE
        stepLabelText.text = "1 / 4 · 시작 안내"
        titleText.text = "VIA 사용 전 안내"
        bodyText.text = "VIA는 보행자 신호와 주변 횡단보도 정보를 보조적으로 안내합니다."
        detailText.text = "앱은 최종 판단을 대신하지 않습니다. 실제 이동 전에는 차량, 자전거, 주변 사람, 노면 상태를 직접 확인해 주세요."
        actionButton.isEnabled = true
        actionButton.text = "다음"
        secondaryButton.visibility = View.VISIBLE
        secondaryButton.text = "다시 듣기"
        replayButton.visibility = View.GONE
        applyActionButtonDensity(infoMode = true)
        speakCurrentStep(forceReplay = true)
    }

    private fun showPermissionStep(detailOverride: String? = null) {
        currentStep = Step.PERMISSION
        applyLayoutMode(infoMode = true)
        calibrationPreview.visibility = View.GONE
        previewCard.visibility = View.GONE
        progressBar.visibility = View.GONE
        progressText.visibility = View.GONE
        stepLabelText.text = "2 / 4 · 권한 허용"
        titleText.text = "권한이 필요합니다"
        bodyText.text = "카메라는 보행자 신호 확인, 위치는 주변 횡단보도 안내에 사용합니다."
        detailText.text = detailOverride ?: "SMS 권한은 등록한 보호자나 기관에 비상 문자를 자동 발송할 때만 사용합니다."
        actionButton.isEnabled = true
        actionButton.text = "권한 허용"
        secondaryButton.visibility = View.VISIBLE
        secondaryButton.text = "다시 듣기"
        replayButton.visibility = View.GONE
        applyActionButtonDensity(infoMode = true)
        speakCurrentStep(forceReplay = true)
    }

    private fun requestOnboardingPermissions() {
        val missingPermissions =
            OnboardingPermissionPolicy.missingPermissions(
                hasCameraPermission = hasCameraPermission(),
                hasLocationPermission = hasLocationPermission(),
                hasSmsPermission = hasSmsPermission()
            )

        if (missingPermissions.isEmpty()) {
            showEmergencyContactStep()
            return
        }

        permissionLauncher.launch(missingPermissions.toTypedArray())
    }

    private fun showEmergencyContactStep() {
        currentStep = Step.EMERGENCY_CONTACT
        applyLayoutMode(infoMode = true)
        calibrationPreview.visibility = View.GONE
        previewCard.visibility = View.GONE
        progressBar.visibility = View.GONE
        progressText.visibility = View.GONE
        stepLabelText.text = "3 / 4 · 비상 연락"
        titleText.text = "보호자 연락처를 준비하세요"
        updateEmergencyContactStep()
        replayButton.visibility = View.GONE
        applyActionButtonDensity(infoMode = true)
        speakCurrentStep(forceReplay = true)
    }

    private fun updateEmergencyContactStep() {
        if (!::bodyText.isInitialized || currentStep != Step.EMERGENCY_CONTACT) {
            return
        }
        val savedPhone = preferences.emergencyContactPhone
        bodyText.text =
            if (savedPhone.isNullOrBlank()) {
                "비상 상황에서 보낼 연락처를 미리 등록해 두면 필요할 때 빠르게 비상 문자를 보낼 수 있습니다."
            } else {
                "등록된 비상 연락처가 있습니다."
            }
        detailText.text =
            if (savedPhone.isNullOrBlank()) {
                "연락처 앱에서 보호자나 기관 번호를 선택할 수 있습니다. 지금 등록하지 않아도 나중에 설정에서 추가할 수 있습니다."
            } else {
                "${preferences.emergencyContactName ?: "비상 연락처"} · $savedPhone"
            }
        actionButton.isEnabled = true
        actionButton.text = if (savedPhone.isNullOrBlank()) "연락처 설정하기" else "다음"
        secondaryButton.visibility = View.VISIBLE
        secondaryButton.text = "나중에 하기"
        applyActionButtonDensity(infoMode = true)
    }

    private fun hasEmergencyContact(): Boolean =
        !preferences.emergencyContactPhone.isNullOrBlank()

    private fun openEmergencyContactSetup(markOnboardingComplete: Boolean) {
        if (markOnboardingComplete) {
            preferences.onboardingCompleted = true
        }
        startActivity(Intent(this, EmergencyContactActivity::class.java))
    }

    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
    }

    private fun hasLocationPermission(): Boolean {
        val hasFine =
            ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) ==
                PackageManager.PERMISSION_GRANTED
        val hasCoarse =
            ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) ==
                PackageManager.PERMISSION_GRANTED
        return hasFine || hasCoarse
    }

    private fun hasSmsPermission(): Boolean {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.SEND_SMS) ==
            PackageManager.PERMISSION_GRANTED
    }

    private fun startCalibration() {
        if (cameraExecutor == null) {
            cameraExecutor = Executors.newSingleThreadExecutor()
        }
        currentStep = Step.CALIBRATING
        applyLayoutMode(infoMode = false)
        calibrationFinalized = false
        calibrationResults.clear()
        currentCandidateIndex = -1
        reusableRotatedBitmap = null
        candidateProfiles = CalibrationSelector.calibrationCandidates(discoverModelFiles())
        calibrationPreview.visibility = View.VISIBLE
        previewCard.visibility = View.VISIBLE
        progressBar.visibility = View.VISIBLE
        progressText.visibility = View.VISIBLE
        progressBar.max = (WARMUP_MS + MEASURE_MS).toInt()
        progressBar.progress = 0
        stepLabelText.text = "4 / 4 · 자동 최적화"
        titleText.text = "설정을 확인하는 중입니다"
        bodyText.text = "이 기기에서 사용할 AI 모델과 실행 방식을 짧게 측정합니다."
        detailText.text = "휴대폰을 안정적으로 들고 잠시만 기다려 주세요."
        actionButton.isEnabled = false
        actionButton.text = "측정 중"
        secondaryButton.visibility = View.VISIBLE
        secondaryButton.text = "다시 듣기"
        replayButton.visibility = View.GONE
        applyActionButtonDensity(infoMode = false)
        speakCurrentStep(forceReplay = true)
        advanceCalibrationCandidate()
    }

    private fun discoverModelFiles(): List<String> {
        return assets.list("")
            ?.let { DetectionLabels.modelFilesForActiveSchema(it.toList()) }
            ?: emptyList()
    }

    private fun advanceCalibrationCandidate() {
        cameraManager?.stopCamera()
        calibrationDetector?.close()
        calibrationDetector = null
        activeRun = null

        currentCandidateIndex += 1
        if (currentCandidateIndex >= candidateProfiles.size) {
            finalizeCalibration()
            return
        }

        val profile = candidateProfiles[currentCandidateIndex]
        progressBar.progress = 0
        progressText.text = "${currentCandidateIndex + 1} / ${candidateProfiles.size}"
        bodyText.text = "${profile.fileName} 측정 중"
        detailText.text = "목표: 15 FPS 이상"

        val candidateUsesGpu = profile.recommendedUseGpu
        val executor = cameraExecutor ?: return
        executor.execute {
            try {
                val detector = YoloDetector(
                    context = this,
                    modelPath = profile.fileName,
                    useGpu = candidateUsesGpu,
                    labels = DetectionLabels.sevenClassLabels,
                    defaultIouThreshold = 0.5f,
                    specificIouThresholds =
                        mapOf(
                            DetectionLabels.HUMAN_RED to 0.05f,
                            DetectionLabels.HUMAN_GREEN to 0.05f,
                            DetectionLabels.VEHICLE_RED to 0.05f,
                            DetectionLabels.VEHICLE_GREEN to 0.05f
                        )
                )
                detector.setup()
                calibrationDetector = detector
                activeRun = ActiveCalibrationRun(profile, detector.runtimeBackendLabel, detector.compatibilityReportedSupported)
                runOnUiThread {
                    bodyText.text = "${profile.fileName} 측정 중"
                    detailText.text = "${detector.runtimeBackendLabel} 가속"
                    bindCalibrationCamera(profile)
                }
            } catch (_: Exception) {
                calibrationResults += CalibrationProfileResult(
                    profile = profile,
                    backendLabel = if (candidateUsesGpu) "GPU unavailable" else "CPU unavailable",
                    averageInputFps = 0.0,
                    averageDetectFps = 0.0,
                    averageDetectLatencyMs = 0L,
                    averageTotalLatencyMs = 0L,
                    compatibilityReportedSupported = false,
                    isUsable = false
                )
                runOnUiThread { advanceCalibrationCandidate() }
            }
        }
    }

    private fun bindCalibrationCamera(profile: InferenceModelProfile) {
        val executor = cameraExecutor ?: return
        cameraManager = CameraManager(
            context = this,
            lifecycleOwner = this,
            viewFinder = calibrationPreview,
            executor = executor,
            analysisTargetResolution = android.util.Size(
                profile.analysisResolution.width,
                profile.analysisResolution.height
            )
        ) { image ->
            processCalibrationFrame(image)
        }
        cameraManager?.startCamera()
    }

    private fun processCalibrationFrame(imageProxy: ImageProxy) {
        val detector = calibrationDetector
        val run = activeRun
        if (detector == null || run == null || run.finished) {
            imageProxy.close()
            return
        }

        val frameStartNs = SystemClock.elapsedRealtimeNanos()
        var imageClosed = false
        try {
            val bitmap = imageProxy.toBitmap()
            val rotationDegrees = imageProxy.imageInfo.rotationDegrees
            imageProxy.close()
            imageClosed = true
            val rotatedBitmap = ImageUtils.rotateBitmap(
                bitmap = bitmap,
                degrees = rotationDegrees.toFloat(),
                reusableBitmap = reusableRotatedBitmap
            )
            if (rotatedBitmap !== bitmap) {
                reusableRotatedBitmap = rotatedBitmap
            }

            val detectResult = detector.detect(rotatedBitmap, 0.15f)
            val elapsedSinceStartMs = SystemClock.elapsedRealtime() - run.startedAtMs
            runOnUiThread {
                progressBar.progress = elapsedSinceStartMs.coerceAtMost(WARMUP_MS + MEASURE_MS).toInt()
            }
            if (elapsedSinceStartMs >= WARMUP_MS) {
                run.measuredInputFrames += 1
                run.measuredDetectFrames += 1
                run.totalDetectLatencyMs += detectResult.inferenceTime
                run.totalPipelineLatencyMs += elapsedMillis(frameStartNs)
            }

            if (elapsedSinceStartMs >= (WARMUP_MS + MEASURE_MS) && !run.finished) {
                run.finished = true
                val result = run.toResult()
                calibrationResults += result
                runOnUiThread {
                    bodyText.text = "${result.profile.fileName} 측정 완료"
                    detailText.text = "${"%.1f".format(result.averageDetectFps)} FPS"
                    if (result.meetsTarget()) {
                        finalizeCalibration()
                    } else {
                        advanceCalibrationCandidate()
                    }
                }
            }
        } catch (_: Exception) {
            if (!run.finished) {
                run.finished = true
                calibrationResults += CalibrationProfileResult(
                    profile = run.profile,
                    backendLabel = "${run.backendLabel} failed",
                    averageInputFps = 0.0,
                    averageDetectFps = 0.0,
                    averageDetectLatencyMs = 0L,
                    averageTotalLatencyMs = 0L,
                    compatibilityReportedSupported = run.compatibilityReportedSupported,
                    isUsable = false
                )
                runOnUiThread { advanceCalibrationCandidate() }
            }
        } finally {
            if (!imageClosed) {
                imageProxy.close()
            }
        }
    }

    private fun finalizeCalibration() {
        if (calibrationFinalized) return
        calibrationFinalized = true
        applyLayoutMode(infoMode = true)
        cameraManager?.stopCamera()
        calibrationDetector?.close()
        calibrationDetector = null
        val bestResult = CalibrationSelector.chooseBest(calibrationResults)
        if (bestResult == null) {
            showPermissionStep(detailOverride = "측정에 실패했습니다. 다시 권한을 확인하고 시도해 주세요.")
            return
        }

        val summaryLines = calibrationResults.joinToString("\n") { result ->
            "${result.profile.fileName} / ${result.backendLabel} / ${"%.1f".format(result.averageDetectFps)} FPS"
        }
        preferences.saveCalibration(
            result = bestResult,
            deviceSummary = deviceSummary,
            summary = summaryLines
        )

        currentStep = Step.RESULT
        calibrationPreview.visibility = View.GONE
        previewCard.visibility = View.GONE
        progressBar.visibility = View.GONE
        progressText.visibility = View.GONE
        stepLabelText.text = "완료"
        titleText.text = "설정이 완료되었습니다"
        bodyText.text = bestResult.profile.fileName
        detailText.text = "${bestResult.backendLabel} / ${"%.1f".format(bestResult.averageDetectFps)} FPS · 보호자 연락처는 설정에서 다시 수정할 수 있습니다."
        actionButton.isEnabled = true
        actionButton.text = "시작하기"
        secondaryButton.visibility = View.VISIBLE
        secondaryButton.text = "비상 연락처 설정"
        replayButton.visibility = View.VISIBLE
        replayButton.text = "다시 듣기"
        applyActionButtonDensity(infoMode = true)
        speakCurrentStep(forceReplay = true)
    }

    private fun speakCurrentStep(forceReplay: Boolean = false) {
        if (!ttsReady) return
        val message = when (currentStep) {
            Step.INTRO ->
                "VIA는 보행자 신호와 주변 횡단보도 정보를 보조적으로 안내합니다. 실제 이동 전에는 주변 상황을 직접 확인해 주세요."
            Step.PERMISSION ->
                "카메라는 보행자 신호 확인에, 위치는 주변 횡단보도 안내에 사용합니다. SMS 권한은 비상 문자 자동 발송에만 사용합니다."
            Step.EMERGENCY_CONTACT ->
                if (hasEmergencyContact()) {
                    "비상 연락처가 등록되어 있습니다. 다음 단계로 진행할 수 있습니다."
                } else {
                    "보호자나 기관 연락처를 미리 등록해 두면 필요할 때 비상 문자를 빠르게 보낼 수 있습니다."
                }
            Step.CALIBRATING ->
                "기기에서 사용할 AI 모델과 실행 방식을 측정하는 중입니다. 잠시만 기다려 주세요."
            Step.RESULT ->
                "설정이 완료되었습니다. 보호자 연락처는 설정에서 다시 수정할 수 있습니다."
        }
        if (!forceReplay && spokenMessage == message) return
        spokenMessage = message
        tts.speak(message, TextToSpeech.QUEUE_FLUSH, null, "onboarding_step")
    }

    private fun applySystemBarInsets() {
        ViewCompat.setOnApplyWindowInsetsListener(rootView) { _, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            updateTopMargin(buildInfoCard, bars.top + 28)
            updateTopMargin(headerCard, 24)
            actionContainer.setPadding(
                actionContainer.paddingLeft,
                actionContainer.paddingTop,
                actionContainer.paddingRight,
                bars.bottom + 16
            )
            insets
        }
    }

    private fun updateTopMargin(view: View, topMargin: Int) {
        val layoutParams = view.layoutParams as? ViewGroup.MarginLayoutParams ?: return
        layoutParams.topMargin = topMargin
        view.layoutParams = layoutParams
    }

    private fun applyLayoutMode(infoMode: Boolean) {
        val layoutParams = actionContainer.layoutParams as? ConstraintLayout.LayoutParams ?: return
        if (infoMode) {
            layoutParams.topToBottom = R.id.onboardingHeaderCard
            layoutParams.bottomToBottom = ConstraintLayout.LayoutParams.PARENT_ID
            layoutParams.height = 0
            layoutParams.topMargin = 8
            layoutParams.bottomMargin = 16
        } else {
            layoutParams.topToBottom = ConstraintLayout.LayoutParams.UNSET
            layoutParams.bottomToBottom = ConstraintLayout.LayoutParams.PARENT_ID
            layoutParams.height = ViewGroup.LayoutParams.WRAP_CONTENT
            layoutParams.topMargin = 0
            layoutParams.bottomMargin = 24
        }
        actionContainer.layoutParams = layoutParams
        applyActionButtonDensity(infoMode)
    }

    private fun applyActionButtonDensity(infoMode: Boolean) {
        val visibleButtons = listOf(actionButton, secondaryButton, replayButton).filter { it.visibility == View.VISIBLE }
        visibleButtons.forEachIndexed { index, button ->
            val params = button.layoutParams as? LinearLayout.LayoutParams ?: return@forEachIndexed
            params.width = ViewGroup.LayoutParams.MATCH_PARENT
            params.topMargin = if (index == 0) 0 else 12
            if (infoMode) {
                params.height = 0
                params.weight = 1f
            } else {
                params.height = ViewGroup.LayoutParams.WRAP_CONTENT
                params.weight = 0f
            }
            button.layoutParams = params
            button.minHeight = if (infoMode) dp(if (button === actionButton) 96 else 84) else dp(if (button === actionButton) 72 else 64)
        }
    }

    private fun dp(value: Int): Int =
        (value * resources.displayMetrics.density).toInt()

    private fun launchMain() {
        activeRun?.finished = true
        cameraManager?.stopCamera()
        cameraManager = null
        calibrationDetector?.close()
        calibrationDetector = null
        reusableRotatedBitmap = null
        cameraExecutor?.shutdownNow()
        cameraExecutor = null
        KineticGuestSessionManager.from(this).prefetchIfNeeded()
        startActivity(Intent(this, MainActivity::class.java))
        finish()
    }

    override fun onDestroy() {
        super.onDestroy()
        activeRun?.finished = true
        cameraManager?.stopCamera()
        cameraManager = null
        calibrationDetector?.close()
        calibrationDetector = null
        cameraExecutor?.shutdown()
        cameraExecutor = null
        if (::tts.isInitialized) {
            tts.stop()
            tts.shutdown()
        }
    }

    private fun elapsedMillis(startTimeNs: Long): Long =
        (SystemClock.elapsedRealtimeNanos() - startTimeNs) / 1_000_000L

    private class ActiveCalibrationRun(
        val profile: InferenceModelProfile,
        val backendLabel: String,
        val compatibilityReportedSupported: Boolean,
        val startedAtMs: Long = SystemClock.elapsedRealtime()
    ) {
        var measuredInputFrames: Int = 0
        var measuredDetectFrames: Int = 0
        var totalDetectLatencyMs: Long = 0
        var totalPipelineLatencyMs: Long = 0
        var finished: Boolean = false

        fun toResult(): CalibrationProfileResult {
            val durationSeconds = MEASURE_MS / 1000.0
            val averageDetectLatency = if (measuredDetectFrames == 0) 0L else totalDetectLatencyMs / measuredDetectFrames
            val averageTotalLatency = if (measuredDetectFrames == 0) 0L else totalPipelineLatencyMs / measuredDetectFrames
            return CalibrationProfileResult(
                profile = profile,
                backendLabel = backendLabel,
                averageInputFps = measuredInputFrames / durationSeconds,
                averageDetectFps = measuredDetectFrames / durationSeconds,
                averageDetectLatencyMs = averageDetectLatency,
                averageTotalLatencyMs = averageTotalLatency,
                compatibilityReportedSupported = compatibilityReportedSupported,
                isUsable = true
            )
        }
    }

    companion object {
        private const val WARMUP_MS = 2_000L
        private const val MEASURE_MS = 5_000L
    }
}
