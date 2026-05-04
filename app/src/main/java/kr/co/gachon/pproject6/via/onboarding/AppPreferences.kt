package kr.co.gachon.pproject6.via.onboarding

import android.content.Context

class AppPreferences(context: Context) {
    private val prefs = context.applicationContext.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)

    var onboardingCompleted: Boolean
        get() = prefs.getBoolean(KEY_ONBOARDING_COMPLETED, false)
        set(value) = prefs.edit().putBoolean(KEY_ONBOARDING_COMPLETED, value).apply()

    var selectedModelName: String?
        get() = prefs.getString(KEY_SELECTED_MODEL_NAME, null)
        set(value) = prefs.edit().putString(KEY_SELECTED_MODEL_NAME, value).apply()

    var selectedBackendLabel: String?
        get() = prefs.getString(KEY_SELECTED_BACKEND_LABEL, null)
        set(value) = prefs.edit().putString(KEY_SELECTED_BACKEND_LABEL, value).apply()

    var calibrationSummary: String?
        get() = prefs.getString(KEY_CALIBRATION_SUMMARY, null)
        set(value) = prefs.edit().putString(KEY_CALIBRATION_SUMMARY, value).apply()

    var deviceSummary: String?
        get() = prefs.getString(KEY_DEVICE_SUMMARY, null)
        set(value) = prefs.edit().putString(KEY_DEVICE_SUMMARY, value).apply()

    var calibrationCompletedAtMillis: Long
        get() = prefs.getLong(KEY_CALIBRATION_COMPLETED_AT, 0L)
        set(value) = prefs.edit().putLong(KEY_CALIBRATION_COMPLETED_AT, value).apply()

    var mapDatasetVersion: String?
        get() = prefs.getString(KEY_MAP_DATASET_VERSION, null)
        set(value) = prefs.edit().putString(KEY_MAP_DATASET_VERSION, value).apply()

    var mapLastDatasetCheckAtMillis: Long
        get() = prefs.getLong(KEY_MAP_LAST_DATASET_CHECK_AT, 0L)
        set(value) = prefs.edit().putLong(KEY_MAP_LAST_DATASET_CHECK_AT, value).apply()

    var mapInstallationId: String?
        get() = prefs.getString(KEY_MAP_INSTALLATION_ID, null)
        set(value) = prefs.edit().putString(KEY_MAP_INSTALLATION_ID, value).apply()

    var voiceGuidanceEnabled: Boolean
        get() = prefs.getBoolean(KEY_VOICE_GUIDANCE_ENABLED, true)
        set(value) = prefs.edit().putBoolean(KEY_VOICE_GUIDANCE_ENABLED, value).apply()

    var hapticFeedbackEnabled: Boolean
        get() = prefs.getBoolean(KEY_HAPTIC_FEEDBACK_ENABLED, true)
        set(value) = prefs.edit().putBoolean(KEY_HAPTIC_FEEDBACK_ENABLED, value).apply()

    var screenColorFeedbackEnabled: Boolean
        get() = prefs.getBoolean(KEY_SCREEN_COLOR_FEEDBACK_ENABLED, true)
        set(value) = prefs.edit().putBoolean(KEY_SCREEN_COLOR_FEEDBACK_ENABLED, value).apply()

    fun saveCalibration(
        result: CalibrationProfileResult,
        deviceSummary: String,
        summary: String
    ) {
        prefs.edit()
            .putBoolean(KEY_ONBOARDING_COMPLETED, true)
            .putString(KEY_SELECTED_MODEL_NAME, result.profile.fileName)
            .putString(KEY_SELECTED_BACKEND_LABEL, result.backendLabel)
            .putString(KEY_CALIBRATION_SUMMARY, summary)
            .putString(KEY_DEVICE_SUMMARY, deviceSummary)
            .putLong(KEY_CALIBRATION_COMPLETED_AT, System.currentTimeMillis())
            .apply()
    }

    fun clearCalibration() {
        prefs.edit()
            .remove(KEY_SELECTED_MODEL_NAME)
            .remove(KEY_SELECTED_BACKEND_LABEL)
            .remove(KEY_CALIBRATION_SUMMARY)
            .remove(KEY_DEVICE_SUMMARY)
            .remove(KEY_CALIBRATION_COMPLETED_AT)
            .putBoolean(KEY_ONBOARDING_COMPLETED, false)
            .apply()
    }

    companion object {
        private const val PREF_NAME = "via_onboarding_preferences"
        private const val KEY_ONBOARDING_COMPLETED = "onboarding_completed"
        private const val KEY_SELECTED_MODEL_NAME = "selected_model_name"
        private const val KEY_SELECTED_BACKEND_LABEL = "selected_backend_label"
        private const val KEY_CALIBRATION_SUMMARY = "calibration_summary"
        private const val KEY_DEVICE_SUMMARY = "device_summary"
        private const val KEY_CALIBRATION_COMPLETED_AT = "calibration_completed_at"
        private const val KEY_MAP_DATASET_VERSION = "map_dataset_version"
        private const val KEY_MAP_LAST_DATASET_CHECK_AT = "map_last_dataset_check_at"
        private const val KEY_MAP_INSTALLATION_ID = "map_installation_id"
        private const val KEY_VOICE_GUIDANCE_ENABLED = "voice_guidance_enabled"
        private const val KEY_HAPTIC_FEEDBACK_ENABLED = "haptic_feedback_enabled"
        private const val KEY_SCREEN_COLOR_FEEDBACK_ENABLED = "screen_color_feedback_enabled"
    }
}
