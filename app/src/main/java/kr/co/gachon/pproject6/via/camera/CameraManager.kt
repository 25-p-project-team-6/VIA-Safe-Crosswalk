package kr.co.gachon.pproject6.via.camera

import android.content.Context
import android.util.Size
import android.util.Log
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
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
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null

    fun startCamera(onZoomStateReady: ((maxZoom: Float) -> Unit)? = null) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            this.cameraProvider = cameraProvider

            val preview = Preview.Builder()
                .build()
                .also {
                    it.surfaceProvider = viewFinder.surfaceProvider
                }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetResolution(analysisTargetResolution)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also {
                    it.setAnalyzer(executor) { image ->
                        imageAnalyzerCallback(image)
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                camera = cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    cameraSelector,
                    preview,
                    imageAnalyzer
                )

                // Check Zoom capabilities if callback provided
                val zoomState = camera?.cameraInfo?.zoomState?.value
                if (zoomState != null) {
                    onZoomStateReady?.invoke(zoomState.maxZoomRatio)
                }

            } catch (exc: Exception) {
                Log.e("CameraManager", "Use case binding failed", exc)
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
}
