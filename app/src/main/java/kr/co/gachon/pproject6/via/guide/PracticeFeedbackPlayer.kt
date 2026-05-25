package kr.co.gachon.pproject6.via.guide

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.speech.tts.TextToSpeech
import java.util.Locale
import kr.co.gachon.pproject6.via.feedback.SignalFeedbackPatterns

class PracticeFeedbackPlayer(context: Context) : TextToSpeech.OnInitListener {
    private val appContext = context.applicationContext
    private val tts = TextToSpeech(appContext, this)
    private val vibrator: Vibrator? =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            appContext.getSystemService(VibratorManager::class.java)?.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            appContext.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
        }

    private var ttsReady = false

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

    fun play(example: PracticeFeedbackExample): Boolean {
        if (!example.simulationOnly || !ttsReady) {
            return false
        }

        stop()
        speak(example.speechText, "practice_${example.id}")
        example.hapticPattern?.let { vibrate(SignalFeedbackPatterns.copyOf(it)) }
        return true
    }

    fun stop() {
        tts.stop()
        cancelVibration()
    }

    fun release() {
        stop()
        tts.shutdown()
    }

    private fun speak(message: String, utteranceId: String) {
        tts.speak(message, TextToSpeech.QUEUE_FLUSH, null, utteranceId)
    }

    private fun vibrate(pattern: LongArray) {
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
