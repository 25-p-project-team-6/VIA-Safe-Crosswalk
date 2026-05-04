package kr.co.gachon.pproject6.via.feedback

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.speech.tts.TextToSpeech
import java.util.Locale
import kr.co.gachon.pproject6.via.ml.AdvisoryAssessment
import kr.co.gachon.pproject6.via.ml.AdvisoryState
import kr.co.gachon.pproject6.via.ml.GuidanceTuningDefaults

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

    fun onAdvisoryChanged(
        assessment: AdvisoryAssessment
    ) {
        val family =
            when (assessment.state) {
                AdvisoryState.RED_CONFIRMED,
                AdvisoryState.GREEN_CONFIRMED,
                AdvisoryState.GREEN_WITH_CAUTION -> FeedbackRepeatFamily.ACTION_LIKE
                AdvisoryState.TRANSITION_WAIT,
                AdvisoryState.UNCERTAIN_VIEW -> FeedbackRepeatFamily.WAIT_LIKE
            }
        if (!feedbackPolicy.shouldEmit(assessment.speechText, family)) {
            return
        }

        when (assessment.state) {
            AdvisoryState.RED_CONFIRMED -> {
                speak(assessment.speechText)
                vibrate(longArrayOf(0, 400, 200, 400))
            }

            AdvisoryState.GREEN_CONFIRMED -> {
                speak(assessment.speechText)
                vibrate(longArrayOf(0, 180, 120, 180, 120, 180))
            }

            AdvisoryState.GREEN_WITH_CAUTION -> {
                speak(assessment.speechText)
                vibrate(longArrayOf(0, 150, 100, 150, 250, 150))
            }

            AdvisoryState.TRANSITION_WAIT,
            AdvisoryState.UNCERTAIN_VIEW -> {
                speak(assessment.speechText)
                vibrate(longArrayOf(0, 140))
            }
        }
    }

    fun clearState() {
        feedbackPolicy.clear()
        tts.stop()
        cancelVibration()
    }

    fun speakImmediate(message: String, utteranceId: String = "manual_guidance") {
        speak(message, utteranceId)
    }

    fun release() {
        clearState()
        tts.shutdown()
    }

    private fun speak(message: String, utteranceId: String = "traffic_light_state") {
        if (!voiceEnabled || !ttsReady) {
            return
        }

        tts.speak(message, TextToSpeech.QUEUE_FLUSH, null, utteranceId)
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
