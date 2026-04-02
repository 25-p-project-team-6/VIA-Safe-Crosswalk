package kr.co.gachon.pproject6.via.feedback

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.speech.tts.TextToSpeech
import java.util.Locale
import kr.co.gachon.pproject6.via.ml.GuidanceTuningDefaults
import kr.co.gachon.pproject6.via.ml.UserGuidanceState

class SignalFeedbackManager(context: Context) : TextToSpeech.OnInitListener {
    private val appContext = context.applicationContext
    private val tts = TextToSpeech(appContext, this)
    private val feedbackPolicy =
        SignalFeedbackPolicy(timingConfig = GuidanceTuningDefaults.feedbackTimingConfig)
    private val vibrator: Vibrator? =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            appContext.getSystemService(VibratorManager::class.java)?.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            appContext.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
        }

    private var ttsReady = false
    var voiceEnabled = true
    var hapticEnabled = true

    override fun onInit(status: Int) {
        if (status != TextToSpeech.SUCCESS) {
            return
        }

        val preferred = tts.setLanguage(Locale.KOREAN)
        if (preferred == TextToSpeech.LANG_MISSING_DATA || preferred == TextToSpeech.LANG_NOT_SUPPORTED) {
            tts.setLanguage(Locale.getDefault())
        }
        ttsReady = true
    }

    fun onGuidanceStateChanged(
        state: UserGuidanceState,
        occupancyCaution: Boolean = false
    ) {
        if (!feedbackPolicy.shouldEmit(state, occupancyCaution)) {
            return
        }

        when (state) {
            UserGuidanceState.STOP -> {
                speak("멈추세요")
                vibrate(longArrayOf(0, 400, 200, 400))
            }

            UserGuidanceState.GO -> {
                if (occupancyCaution) {
                    speak("건너세요. 차량 주의.")
                    vibrate(longArrayOf(0, 150, 100, 150, 250, 150))
                } else {
                    speak("건너세요")
                    vibrate(longArrayOf(0, 180, 120, 180, 120, 180))
                }
            }

            UserGuidanceState.WAIT -> {
                speak("잠시 기다리세요")
                vibrate(longArrayOf(0, 140))
            }
        }
    }

    fun clearState() {
        feedbackPolicy.clear()
        tts.stop()
        cancelVibration()
    }

    fun release() {
        clearState()
        tts.shutdown()
    }

    private fun speak(message: String) {
        if (!voiceEnabled || !ttsReady) {
            return
        }

        tts.speak(message, TextToSpeech.QUEUE_FLUSH, null, "traffic_light_state")
    }

    private fun vibrate(pattern: LongArray) {
        if (!hapticEnabled) {
            return
        }

        val targetVibrator = vibrator
        if (targetVibrator == null || !targetVibrator.hasVibrator()) {
            return
        }

        cancelVibration()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            targetVibrator.vibrate(VibrationEffect.createWaveform(pattern, -1))
        } else {
            @Suppress("DEPRECATION")
            targetVibrator.vibrate(pattern, -1)
        }
    }

    private fun cancelVibration() {
        val targetVibrator = vibrator
        if (targetVibrator == null || !targetVibrator.hasVibrator()) {
            return
        }

        targetVibrator.cancel()
    }
}
