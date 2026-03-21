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
import android.widget.ProgressBar
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.ImageProxy
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.google.android.material.button.MaterialButton
import kr.co.gachon.pproject6.via.MainActivity
import kr.co.gachon.pproject6.via.R
import kr.co.gachon.pproject6.via.BuildConfig
import kr.co.gachon.pproject6.via.camera.CameraManager
import kr.co.gachon.pproject6.via.ml.InferenceModelProfile
import kr.co.gachon.pproject6.via.ml.YoloDetector
import kr.co.gachon.pproject6.via.util.ImageUtils
import org.tensorflow.lite.gpu.CompatibilityList
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class OnboardingActivity : AppCompatActivity(), TextToSpeech.OnInitListener {
    private enum class Step {
        INTRO,
        PERMISSION,
        CALIBRATING,
        RESULT
    }

    private lateinit var titleText: TextView
    private lateinit var stepLabelText: TextView
    private lateinit var bodyText: TextView
    private lateinit var detailText: TextView
    private lateinit var actionButton: MaterialButton
    private lateinit var secondaryButton: MaterialButton
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
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                startCalibration()
            } else {
                showPermissionStep(
                    detailOverride = "카메라 권한이 없으면 신호를 볼 수 없습니다. 허용 버튼을 눌러 주세요."
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
        progressBar = findViewById(R.id.onboardingProgressBar)
        progressText = findViewById(R.id.onboardingProgressText)
        calibrationPreview = findViewById(R.id.onboardingPreview)
        buildInfoText.text = "v${BuildConfig.VERSION_NAME} (${BuildConfig.VERSION_CODE}) · ${BuildConfig.BUILD_STAMP}"
        cameraExecutor = Executors.newSingleThreadExecutor()
        tts = TextToSpeech(this, this)

        actionButton.setOnClickListener { handlePrimaryAction() }
        secondaryButton.setOnClickListener { handleSecondaryAction() }
        applySystemBarInsets()

        showIntroStep()
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
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    startCalibration()
                } else {
                    permissionLauncher.launch(Manifest.permission.CAMERA)
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
            Step.RESULT -> {
                preferences.clearCalibration()
                startCalibration()
            }
            else -> speakCurrentStep(forceReplay = true)
        }
    }

    private fun showIntroStep() {
        currentStep = Step.INTRO
        calibrationPreview.visibility = View.GONE
        previewCard.visibility = View.GONE
        progressBar.visibility = View.GONE
        progressText.visibility = View.GONE
        stepLabelText.text = "1 / 3 · 시작 안내"
        titleText.text = "처음 설정을 시작합니다"
        bodyText.text = "이 앱은 횡단보도 신호를 음성으로 안내합니다."
        detailText.text = "시각장애인을 위해 음성 중심으로 안내하고, 지금부터 휴대폰 성능을 자동으로 측정합니다."
        actionButton.isEnabled = true
        actionButton.text = "다음"
        secondaryButton.visibility = View.VISIBLE
        secondaryButton.text = "다시 듣기"
        speakCurrentStep(forceReplay = true)
    }

    private fun showPermissionStep(detailOverride: String? = null) {
        currentStep = Step.PERMISSION
        calibrationPreview.visibility = View.GONE
        previewCard.visibility = View.GONE
        progressBar.visibility = View.GONE
        progressText.visibility = View.GONE
        stepLabelText.text = "2 / 3 · 권한 허용"
        titleText.text = "카메라 권한이 필요합니다"
        bodyText.text = "신호등을 보기 위해 카메라 권한을 요청합니다."
        detailText.text = detailOverride ?: "허용 버튼을 누르면 바로 자동 최적화를 시작합니다."
        actionButton.isEnabled = true
        actionButton.text = "권한 허용"
        secondaryButton.visibility = View.VISIBLE
        secondaryButton.text = "다시 듣기"
        speakCurrentStep(forceReplay = true)
    }

    private fun startCalibration() {
        if (cameraExecutor == null) {
            cameraExecutor = Executors.newSingleThreadExecutor()
        }
        currentStep = Step.CALIBRATING
        calibrationFinalized = false
        calibrationResults.clear()
        currentCandidateIndex = -1
        reusableRotatedBitmap = null
        candidateProfiles = CalibrationSelector.calibrationCandidates(discoverModelFiles())
        calibrationPreview.visibility = View.VISIBLE
        previewCard.visibility = View.VISIBLE
        progressBar.visibility = View.VISIBLE
        progressText.visibility = View.VISIBLE
        progressBar.max = candidateProfiles.size.coerceAtLeast(1)
        progressBar.progress = 0
        stepLabelText.text = "3 / 3 · 자동 최적화"
        titleText.text = "휴대폰 성능을 측정하는 중입니다"
        bodyText.text = "기기 모델과 GPU 가능 여부를 확인하고 가장 적절한 모델을 자동으로 고릅니다."
        detailText.text =
            "$deviceSummary\nGPU helper reported: ${CompatibilityList().isDelegateSupportedOnThisDevice}"
        actionButton.isEnabled = false
        actionButton.text = "측정 중"
        secondaryButton.visibility = View.VISIBLE
        secondaryButton.text = "다시 듣기"
        speakCurrentStep(forceReplay = true)
        advanceCalibrationCandidate()
    }

    private fun discoverModelFiles(): List<String> {
        return assets.list("")
            ?.filter { it.endsWith(".tflite", ignoreCase = true) }
            ?.sorted()
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
        progressBar.progress = currentCandidateIndex
        progressText.text = "${currentCandidateIndex + 1} / ${candidateProfiles.size}"
        detailText.text =
            "$deviceSummary\n${profile.displayNameWithSize()} 측정 중\n목표: 15 FPS 이상"

        val candidateUsesGpu = profile.recommendedUseGpu
        val executor = cameraExecutor ?: return
        executor.execute {
            try {
                val detector = YoloDetector(
                    context = this,
                    modelPath = profile.fileName,
                    useGpu = candidateUsesGpu,
                    labels = listOf("bicycle", "car", "motorcycle", "bus", "train", "truck", "green", "red"),
                    defaultIouThreshold = 0.5f,
                    specificIouThresholds = mapOf("red" to 0.05f, "green" to 0.05f)
                )
                detector.setup()
                calibrationDetector = detector
                activeRun = ActiveCalibrationRun(profile, detector.runtimeBackendLabel, detector.compatibilityReportedSupported)
                runOnUiThread {
                    detailText.text =
                        "$deviceSummary\n${profile.displayNameWithSize()} 측정 중\nbackend ${detector.runtimeBackendLabel}"
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
                    progressBar.progress = currentCandidateIndex + 1
                    detailText.text =
                        "$deviceSummary\n${result.profile.displayNameWithSize()}: ${"%.1f".format(result.averageDetectFps)} FPS"
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
        cameraManager?.stopCamera()
        calibrationDetector?.close()
        calibrationDetector = null
        val bestResult = CalibrationSelector.chooseBest(calibrationResults)
        if (bestResult == null) {
            showPermissionStep(detailOverride = "측정에 실패했습니다. 다시 권한을 확인하고 시도해 주세요.")
            return
        }

        val summaryLines = calibrationResults.joinToString("\n") { result ->
            "${result.profile.displayNameWithSize()} / ${result.backendLabel} / ${"%.1f".format(result.averageDetectFps)} FPS"
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
        bodyText.text = buildString {
            append("선택된 설정: ${bestResult.profile.displayNameWithSize()}\n")
            append("백엔드: ${bestResult.backendLabel}\n")
            append("측정 FPS: ${"%.1f".format(bestResult.averageDetectFps)}")
        }
        detailText.text = if (bestResult.meetsTarget()) {
            "15 FPS 이상이 나오는 가장 선명한 모델을 골랐습니다."
        } else {
            "15 FPS를 넘는 모델은 없어 가장 빠른 저속 모드를 골랐습니다."
        }
        actionButton.isEnabled = true
        actionButton.text = "시작하기"
        secondaryButton.visibility = View.VISIBLE
        secondaryButton.text = "다시 측정하기"
        speakCurrentStep(forceReplay = true)
    }

    private fun speakCurrentStep(forceReplay: Boolean = false) {
        if (!ttsReady) return
        val message = when (currentStep) {
            Step.INTRO ->
                "처음 설정을 시작합니다. 이 앱은 횡단보도 신호를 음성으로 안내합니다. 다음 버튼을 눌러 진행해 주세요."
            Step.PERMISSION ->
                "카메라 권한이 필요합니다. 허용 버튼을 누르면 자동 최적화를 시작합니다."
            Step.CALIBRATING ->
                "휴대폰 성능을 측정하는 중입니다. 잠시 기다려 주세요."
            Step.RESULT ->
                "설정이 완료되었습니다. 시작하기 버튼을 누르면 바로 앱을 사용할 수 있습니다. 다시 측정하기 버튼으로 성능 측정을 다시 할 수 있습니다."
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

    private fun launchMain() {
        startActivity(Intent(this, MainActivity::class.java))
        finish()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraManager?.stopCamera()
        calibrationDetector?.close()
        cameraExecutor?.shutdown()
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
