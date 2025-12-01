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
    private lateinit var confidenceSlider: Slider
    private lateinit var gpuSwitch: com.google.android.material.switchmaterial.SwitchMaterial
    private lateinit var debugContainer: android.widget.LinearLayout

    // Set this to false to hide debug info (FPS, Latency, Hardware, Slider)
    private val showDebugInfo = true
    // Set this to false to hide bounding boxes and labels
    private val showBBoxOverlay = false

    private var cameraExecutor: ExecutorService? = null
    @Volatile
    private var detector: YoloDetector? = null
    private var confidenceThreshold = 0.5f

    private var lastFpsTimestamp = System.currentTimeMillis()
    private var frameCount = 0
    
    private var totalFrameCount = 0L
    private var startTime = 0L

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
            if (isGranted) {
                startCamera()
            } else {
                Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
                finish()
            }
        }

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
        debugContainer = findViewById(R.id.debugContainer)
        fpsText = findViewById(R.id.fpsText)
        avgFpsText = findViewById(R.id.avgFpsText)
        latencyText = findViewById(R.id.latencyText)
        confidenceSlider = findViewById(R.id.confidenceSlider)
        gpuSwitch = findViewById(R.id.gpuSwitch)
        
        startTime = System.currentTimeMillis()

        debugContainer.visibility = if (showDebugInfo) android.view.View.VISIBLE else android.view.View.GONE

        confidenceSlider.addOnChangeListener { _, value, _ ->
            confidenceThreshold = value
        }

        gpuSwitch.setOnCheckedChangeListener { _, isChecked ->
            initDetector(isChecked)
            // Reset average stats when hardware changes
            totalFrameCount = 0
            startTime = System.currentTimeMillis()
            avgFpsText.text = "Avg FPS: 0"
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

        frameCount++
        totalFrameCount++
        val currentTime = System.currentTimeMillis()
        val timeDiff = currentTime - lastFpsTimestamp
        
        if (timeDiff >= 1000) {
            val fps = frameCount * 1000.0 / timeDiff
            fpsText.text = String.format("FPS: %.2f", fps)
            frameCount = 0
            lastFpsTimestamp = currentTime
            
            // Update Average FPS
            val totalTimeDiff = currentTime - startTime
            if (totalTimeDiff > 0) {
                val avgFps = totalFrameCount * 1000.0 / totalTimeDiff
                avgFpsText.text = String.format("Avg FPS: %.2f", avgFps)
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
                val newDetector = YoloDetector(this, "yolo11n_float32.tflite", useGpu = useGpu)
                newDetector.setup()
                detector = newDetector
            } catch (e: Exception) {
                Log.e("MainActivity", "Error initializing detector", e)
                runOnUiThread {
                    Toast.makeText(this, "Error initializing detector: ${e.message}", Toast.LENGTH_LONG).show()
                    // Revert switch if failed?
                    if (useGpu) gpuSwitch.isChecked = false
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
                    it.setSurfaceProvider(viewFinder.surfaceProvider)
                }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor!!, { image ->
                        processImage(image)
                    })
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
            if (showBBoxOverlay) {
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