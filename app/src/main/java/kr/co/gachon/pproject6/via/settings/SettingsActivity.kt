package kr.co.gachon.pproject6.via.settings

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.View
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.google.android.material.button.MaterialButton
import com.google.android.material.switchmaterial.SwitchMaterial
import kr.co.gachon.pproject6.via.R
import kr.co.gachon.pproject6.via.map.MapDebugCacheManager
import kr.co.gachon.pproject6.via.onboarding.AppPreferences
import kr.co.gachon.pproject6.via.safety.EmergencyContactActivity

class SettingsActivity : AppCompatActivity() {
    private lateinit var preferences: AppPreferences

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        preferences = AppPreferences(this)
        setContentView(R.layout.activity_settings)
        applySystemBarInsets()

        findViewById<MaterialButton>(R.id.settingsBackButton).setOnClickListener {
            finish()
        }

        bindSwitch(
            switch = findViewById(R.id.voiceGuidanceSwitch),
            initialValue = preferences.voiceGuidanceEnabled,
            onChanged = { preferences.voiceGuidanceEnabled = it }
        )
        bindSwitch(
            switch = findViewById(R.id.hapticFeedbackSwitch),
            initialValue = preferences.hapticFeedbackEnabled,
            onChanged = { preferences.hapticFeedbackEnabled = it }
        )
        bindSwitch(
            switch = findViewById(R.id.screenColorFeedbackSwitch),
            initialValue = preferences.screenColorFeedbackEnabled,
            onChanged = { preferences.screenColorFeedbackEnabled = it }
        )

        findViewById<MaterialButton>(R.id.contactSettingsButton).setOnClickListener {
            startActivity(Intent(this, EmergencyContactActivity::class.java))
        }
        findViewById<MaterialButton>(R.id.openDebugPanelButton).setOnClickListener {
            setResult(
                Activity.RESULT_OK,
                Intent().putExtra(EXTRA_OPEN_DEBUG_PANEL, true)
            )
            finish()
        }
        findViewById<MaterialButton>(R.id.clearMapCacheButton).setOnClickListener {
            val deletedEntries = MapDebugCacheManager.clearAll(this)
            Toast.makeText(this, "지도 캐시 ${deletedEntries}개를 삭제했습니다.", Toast.LENGTH_SHORT).show()
        }

        findViewById<TextView>(R.id.modelSummaryText).text =
            buildString {
                append("현재 AI 모델: ")
                append(preferences.selectedModelName ?: "자동 선택 전")
                preferences.selectedBackendLabel?.let { backend ->
                    append("\n추론 방식: ")
                    append(backend)
                }
            }
        findViewById<TextView>(R.id.permissionStatusText).text = buildPermissionStatusText()
    }

    private fun applySystemBarInsets() {
        val content = findViewById<View>(R.id.settingsContent)
        val baseBottomPadding = content.paddingBottom
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.settingsRoot)) { _, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            content.setPadding(
                content.paddingLeft,
                content.paddingTop,
                content.paddingRight,
                baseBottomPadding + systemBars.bottom
            )
            insets
        }
    }

    private fun bindSwitch(
        switch: SwitchMaterial,
        initialValue: Boolean,
        onChanged: (Boolean) -> Unit
    ) {
        switch.isChecked = initialValue
        switch.setOnCheckedChangeListener { _, isChecked ->
            onChanged(isChecked)
        }
    }

    private fun buildPermissionStatusText(): String {
        val camera = permissionSummary(Manifest.permission.CAMERA)
        val fineLocation = permissionSummary(Manifest.permission.ACCESS_FINE_LOCATION)
        val coarseLocation = permissionSummary(Manifest.permission.ACCESS_COARSE_LOCATION)
        val sms = permissionSummary(Manifest.permission.SEND_SMS)
        val location =
            if (fineLocation == "허용" || coarseLocation == "허용") "허용" else "미허용"
        return "권한 상태: 카메라 $camera · 위치 $location · SMS $sms"
    }

    private fun permissionSummary(permission: String): String {
        return if (ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED) {
            "허용"
        } else {
            "미허용"
        }
    }

    companion object {
        const val EXTRA_OPEN_DEBUG_PANEL = "kr.co.gachon.pproject6.via.OPEN_DEBUG_PANEL"
    }
}
