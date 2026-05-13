package kr.co.gachon.pproject6.via.camera

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.util.Range
import android.util.Log
import android.util.Size
import androidx.annotation.OptIn as AndroidXOptIn
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.CameraInfo
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ExecutorService

class CameraManager(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val viewFinder: PreviewView,
    private val executor: ExecutorService,
    private val analysisTargetResolution: Size,
    private val imageAnalyzerCallback: (ImageProxy) -> Unit
) {
    private companion object {
        private const val TAG = "CameraManager"
    }

    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null

    @AndroidXOptIn(ExperimentalCamera2Interop::class)
    fun startCamera(onZoomStateReady: ((maxZoom: Float) -> Unit)? = null) {
        val viewPort = viewFinder.viewPort
        if (viewPort == null) {
            viewFinder.post { startCamera(onZoomStateReady) }
            return
        }

        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            this.cameraProvider = cameraProvider

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            val flickerMitigationSettings =
                buildFlickerMitigationSettings(cameraProvider, cameraSelector)

            val previewBuilder = Preview.Builder()
            applyFlickerMitigation(previewBuilder, flickerMitigationSettings)
            val preview = previewBuilder
                .build()
                .also {
                    it.surfaceProvider = viewFinder.surfaceProvider
                }

            val imageAnalyzerBuilder = ImageAnalysis.Builder()
                .setTargetResolution(analysisTargetResolution)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            applyFlickerMitigation(imageAnalyzerBuilder, flickerMitigationSettings)
            val imageAnalyzer = imageAnalyzerBuilder
                .build()
                .also {
                    it.setAnalyzer(executor) { image ->
                        imageAnalyzerCallback(image)
                    }
                }

            try {
                cameraProvider.unbindAll()
                val useCaseGroup = UseCaseGroup.Builder()
                    .addUseCase(preview)
                    .addUseCase(imageAnalyzer)
                    .setViewPort(viewPort)
                    .build()
                camera = cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    cameraSelector,
                    useCaseGroup
                )

                // Check Zoom capabilities if callback provided
                val zoomState = camera?.cameraInfo?.zoomState?.value
                if (zoomState != null) {
                    onZoomStateReady?.invoke(zoomState.maxZoomRatio)
                }

            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(context))
    }

    fun setZoom(ratio: Float) {
        camera?.cameraControl?.setZoomRatio(ratio)
    }

    fun stopCamera() {
        cameraProvider?.unbindAll()
        camera = null
    }

    @AndroidXOptIn(ExperimentalCamera2Interop::class)
    private fun buildFlickerMitigationSettings(
        cameraProvider: ProcessCameraProvider,
        cameraSelector: CameraSelector
    ): CameraFlickerMitigationSettings {
        val cameraInfo = selectedCameraInfo(cameraProvider, cameraSelector)
        val camera2Info = cameraInfo?.let { Camera2CameraInfo.from(it) }

        val availableAntibandingModes =
            camera2Info?.getCameraCharacteristic(
                CameraCharacteristics.CONTROL_AE_AVAILABLE_ANTIBANDING_MODES
            )
        val availableFpsRanges =
            camera2Info?.getCameraCharacteristic(
                CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES
            )
                ?.map { CameraFpsRange(lower = it.lower, upper = it.upper) }
                .orEmpty()

        val settings = CameraFlickerMitigationSettings(
            antibandingMode =
                CameraFlickerMitigationPolicy.chooseAntibandingMode(
                    availableModes = availableAntibandingModes,
                    preferredMode = CameraMetadata.CONTROL_AE_ANTIBANDING_MODE_60HZ,
                    autoMode = CameraMetadata.CONTROL_AE_ANTIBANDING_MODE_AUTO,
                    offMode = CameraMetadata.CONTROL_AE_ANTIBANDING_MODE_OFF
                ),
            targetFpsRange =
                CameraFlickerMitigationPolicy.chooseTargetFpsRange(availableFpsRanges)
        )

        if (settings.isEmpty) {
            Log.w(TAG, "No camera flicker mitigation settings available")
        } else {
            Log.i(
                TAG,
                "Camera flicker mitigation: antibanding=${settings.antibandingMode}, " +
                    "fps=${settings.targetFpsRange}"
            )
        }
        return settings
    }

    private fun selectedCameraInfo(
        cameraProvider: ProcessCameraProvider,
        cameraSelector: CameraSelector
    ): CameraInfo? {
        return try {
            cameraSelector
                .filter(cameraProvider.availableCameraInfos)
                .firstOrNull()
        } catch (exc: Exception) {
            Log.w(TAG, "Unable to query selected camera info", exc)
            null
        }
    }

    @AndroidXOptIn(ExperimentalCamera2Interop::class)
    private fun applyFlickerMitigation(
        previewBuilder: Preview.Builder,
        settings: CameraFlickerMitigationSettings
    ) {
        val extender = Camera2Interop.Extender(previewBuilder)
        applyFlickerMitigation(extender, settings)
    }

    @AndroidXOptIn(ExperimentalCamera2Interop::class)
    private fun applyFlickerMitigation(
        imageAnalysisBuilder: ImageAnalysis.Builder,
        settings: CameraFlickerMitigationSettings
    ) {
        val extender = Camera2Interop.Extender(imageAnalysisBuilder)
        applyFlickerMitigation(extender, settings)
    }

    @AndroidXOptIn(ExperimentalCamera2Interop::class)
    private fun <T> applyFlickerMitigation(
        extender: Camera2Interop.Extender<T>,
        settings: CameraFlickerMitigationSettings
    ) {
        settings.antibandingMode?.let { mode ->
            extender.setCaptureRequestOption(
                CaptureRequest.CONTROL_AE_ANTIBANDING_MODE,
                mode
            )
        }
        settings.targetFpsRange?.let { fpsRange ->
            extender.setCaptureRequestOption(
                CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE,
                Range(fpsRange.lower, fpsRange.upper)
            )
        }
    }
}
