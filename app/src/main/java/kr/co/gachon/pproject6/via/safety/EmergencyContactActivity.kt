package kr.co.gachon.pproject6.via.safety

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.CountDownTimer
import android.provider.ContactsContract
import android.telephony.SmsManager
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import kr.co.gachon.pproject6.via.R
import kr.co.gachon.pproject6.via.onboarding.AppPreferences

class EmergencyContactActivity : AppCompatActivity() {
    private lateinit var preferences: AppPreferences
    private lateinit var contactNameInput: EditText
    private lateinit var contactPhoneInput: EditText
    private lateinit var statusText: TextView
    private lateinit var sendButton: MaterialButton
    private lateinit var cancelButton: MaterialButton
    private var countdownTimer: CountDownTimer? = null

    private val requestSmsPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            if (isGranted) {
                startEmergencyCountdown()
            } else {
                Toast.makeText(this, "SMS 권한이 없어 문자 앱을 엽니다.", Toast.LENGTH_LONG).show()
                openSmsApp()
            }
        }

    private val pickContactLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == Activity.RESULT_OK) {
                result.data?.data?.let(::fillContactFromUri)
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        preferences = AppPreferences(this)
        val views = createContentView()
        setContentView(views.root)

        contactNameInput = views.contactNameInput
        contactPhoneInput = views.contactPhoneInput
        statusText = views.statusText
        sendButton = views.sendButton
        cancelButton = views.cancelButton

        contactNameInput.setText(preferences.emergencyContactName.orEmpty())
        contactPhoneInput.setText(preferences.emergencyContactPhone.orEmpty())
        val autoStartCountdown = intent.getBooleanExtra(EXTRA_AUTO_START_COUNTDOWN, false)
        if (autoStartCountdown && !preferences.emergencyContactPhone.isNullOrBlank()) {
            views.contactSettingsCard.visibility = View.GONE
        }
        updateStatus()

        views.backButton.setOnClickListener { finish() }
        views.saveButton.setOnClickListener {
            saveContact()
        }
        views.deleteButton.setOnClickListener {
            preferences.clearEmergencyContact()
            contactNameInput.text.clear()
            contactPhoneInput.text.clear()
            cancelEmergencyCountdown()
            updateStatus()
            Toast.makeText(this, "비상 연락처를 삭제했습니다.", Toast.LENGTH_SHORT).show()
        }
        views.pickContactButton.setOnClickListener {
            openContactPicker()
        }
        sendButton.setOnClickListener {
            requestEmergencyCountdown()
        }
        cancelButton.setOnClickListener {
            cancelEmergencyCountdown()
            statusText.text = "비상 문자 발송을 취소했습니다."
        }
        views.openSmsAppButton.setOnClickListener {
            if (ensureContactSaved()) {
                openSmsApp()
            }
        }
        if (autoStartCountdown) {
            views.root.post {
                requestEmergencyCountdown()
            }
        }
    }

    override fun onStop() {
        cancelEmergencyCountdown()
        super.onStop()
    }

    private fun saveContact(): Boolean {
        val name = contactNameInput.text?.toString()?.trim().orEmpty()
        val phone = contactPhoneInput.text?.toString()?.trim().orEmpty()
        if (phone.isBlank()) {
            contactPhoneInput.error = "전화번호를 입력해 주세요"
            return false
        }

        preferences.emergencyContactName = name.ifBlank { "비상 연락처" }
        preferences.emergencyContactPhone = phone
        updateStatus()
        Toast.makeText(this, "비상 연락처를 저장했습니다.", Toast.LENGTH_SHORT).show()
        return true
    }

    private fun openContactPicker() {
        val intent = Intent(Intent.ACTION_PICK, ContactsContract.CommonDataKinds.Phone.CONTENT_URI)
        runCatching {
            pickContactLauncher.launch(intent)
        }.onFailure {
            Toast.makeText(this, "연락처 앱을 열 수 없습니다.", Toast.LENGTH_LONG).show()
        }
    }

    private fun fillContactFromUri(uri: Uri) {
        val projection =
            arrayOf(
                ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME,
                ContactsContract.CommonDataKinds.Phone.NUMBER
            )
        contentResolver.query(uri, projection, null, null, null)?.use { cursor ->
            if (!cursor.moveToFirst()) return
            val nameIndex = cursor.getColumnIndex(ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME)
            val phoneIndex = cursor.getColumnIndex(ContactsContract.CommonDataKinds.Phone.NUMBER)
            val name =
                if (nameIndex >= 0) cursor.getString(nameIndex).orEmpty() else ""
            val phone =
                if (phoneIndex >= 0) cursor.getString(phoneIndex).orEmpty() else ""
            contactNameInput.setText(name)
            contactPhoneInput.setText(phone)
            if (phone.isNotBlank()) {
                saveContact()
            }
        } ?: Toast.makeText(this, "연락처를 불러오지 못했습니다.", Toast.LENGTH_SHORT).show()
    }

    private fun ensureContactSaved(): Boolean {
        val typedPhone = contactPhoneInput.text?.toString()?.trim().orEmpty()
        val savedPhone = preferences.emergencyContactPhone.orEmpty()
        return if (typedPhone.isNotBlank() && typedPhone != savedPhone) {
            saveContact()
        } else if (savedPhone.isBlank()) {
            statusText.text = "비상 연락처를 먼저 저장해 주세요."
            false
        } else {
            true
        }
    }

    private fun requestEmergencyCountdown() {
        if (!ensureContactSaved()) {
            return
        }
        if (hasSmsPermission()) {
            startEmergencyCountdown()
        } else {
            requestSmsPermissionLauncher.launch(Manifest.permission.SEND_SMS)
        }
    }

    private fun startEmergencyCountdown() {
        val phone = preferences.emergencyContactPhone
        if (phone.isNullOrBlank()) {
            statusText.text = "비상 연락처를 먼저 저장해 주세요."
            return
        }

        cancelEmergencyCountdown()
        sendButton.isEnabled = false
        cancelButton.visibility = View.VISIBLE
        countdownTimer = object : CountDownTimer(AUTO_SEND_DELAY_MS, 1_000L) {
            override fun onTick(millisUntilFinished: Long) {
                val seconds = ((millisUntilFinished + 999L) / 1_000L).coerceAtLeast(1L)
                statusText.text = "${seconds}초 후 보호자에게 비상 문자를 보냅니다. 취소할 수 있습니다."
            }

            override fun onFinish() {
                countdownTimer = null
                cancelButton.visibility = View.GONE
                sendButton.isEnabled = true
                sendEmergencySms(phone)
            }
        }.start()
    }

    @SuppressLint("MissingPermission")
    private fun sendEmergencySms(phone: String) {
        if (!hasSmsPermission()) {
            openSmsApp()
            return
        }

        val message = buildEmergencyMessage()
        runCatching {
            val smsManager =
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                    getSystemService(SmsManager::class.java)
                } else {
                    @Suppress("DEPRECATION")
                    SmsManager.getDefault()
                }
            val parts = smsManager.divideMessage(message)
            smsManager.sendMultipartTextMessage(phone, null, parts, null, null)
        }.onSuccess {
            statusText.text = "비상 문자를 발송했습니다."
            Toast.makeText(this, "비상 문자를 발송했습니다.", Toast.LENGTH_LONG).show()
        }.onFailure {
            statusText.text = "자동 발송에 실패해 문자 앱을 엽니다."
            Toast.makeText(this, "자동 발송 실패: 문자 앱을 엽니다.", Toast.LENGTH_LONG).show()
            openSmsApp()
        }
    }

    private fun openSmsApp() {
        val phone = preferences.emergencyContactPhone
        if (phone.isNullOrBlank()) {
            statusText.text = "비상 연락처를 먼저 저장해 주세요."
            return
        }

        val intent = Intent(Intent.ACTION_SENDTO, Uri.parse("smsto:$phone")).apply {
            putExtra("sms_body", buildEmergencyMessage())
        }
        runCatching {
            startActivity(intent)
        }.onFailure {
            Toast.makeText(this, "문자 앱을 열 수 없습니다.", Toast.LENGTH_LONG).show()
        }
    }

    private fun cancelEmergencyCountdown() {
        countdownTimer?.cancel()
        countdownTimer = null
        cancelButton.visibility = View.GONE
        sendButton.isEnabled = true
    }

    private fun updateStatus() {
        val name = preferences.emergencyContactName
        val phone = preferences.emergencyContactPhone
        statusText.text =
            if (phone.isNullOrBlank()) {
                "연락처를 저장하면 비상 문자 기능을 사용할 수 있습니다."
            } else {
                "등록된 연락처: ${name ?: "비상 연락처"} · $phone"
            }
    }

    private fun buildEmergencyMessage(): String {
        return EmergencyMessageBuilder.build(readLastKnownEmergencyLocation())
    }

    @SuppressLint("MissingPermission")
    private fun readLastKnownEmergencyLocation(): EmergencyLocation? {
        if (!hasLocationPermission()) {
            return null
        }
        val locationManager = getSystemService(LocationManager::class.java) ?: return null
        val location =
            listOf(LocationManager.GPS_PROVIDER, LocationManager.NETWORK_PROVIDER)
                .mapNotNull { provider ->
                    runCatching { locationManager.getLastKnownLocation(provider) }.getOrNull()
                }
                .maxByOrNull(Location::getTime)
                ?: return null
        return EmergencyLocation(location.latitude, location.longitude)
    }

    private fun hasSmsPermission(): Boolean {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.SEND_SMS) ==
            PackageManager.PERMISSION_GRANTED
    }

    private fun hasLocationPermission(): Boolean {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) ==
            PackageManager.PERMISSION_GRANTED ||
            ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) ==
            PackageManager.PERMISSION_GRANTED
    }

    private fun createContentView(): EmergencyContactViews {
        val root = ScrollView(this).apply {
            setBackgroundColor(ContextCompat.getColor(this@EmergencyContactActivity, R.color.via_background))
            isFillViewport = true
        }
        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(28), dp(20), dp(32))
        }
        root.addView(content, ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)

        val backButton = backButton()
        content.addView(backButton, matchWrap())
        content.addView(titleText("비상 연락"), matchWrap(topMargin = 8))
        content.addView(
            bodyText("보호자 또는 기관 연락처를 저장하고, 필요할 때 5초 유예 후 비상 문자를 보냅니다."),
            matchWrap(topMargin = 8)
        )

        val contactNameInput = editText("이름 또는 기관명").apply {
            inputType = android.text.InputType.TYPE_CLASS_TEXT or
                android.text.InputType.TYPE_TEXT_VARIATION_PERSON_NAME
        }
        val contactPhoneInput = editText("전화번호").apply {
            inputType = android.text.InputType.TYPE_CLASS_PHONE
        }
        val pickContactButton = outlinedButton("연락처에서 선택")
        val saveButton = filledButton("연락처 저장")
        val deleteButton = outlinedButton("연락처 삭제")
        val contactSettingsCard =
            card {
                addView(contactNameInput, matchWrap())
                addView(contactPhoneInput, matchWrap(topMargin = 12))
                addView(pickContactButton, matchWrap(topMargin = 12))
                addView(saveButton, matchWrap(topMargin = 18))
                addView(deleteButton, matchWrap(topMargin = 10))
            }
        content.addView(
            contactSettingsCard,
            matchWrap(topMargin = 24)
        )

        val statusText = bodyText("연락처를 저장하면 비상 문자 기능을 사용할 수 있습니다.")
        val sendButton = filledButton("비상 문자 자동 발송").apply {
            minHeight = dp(64)
            textSize = 18f
        }
        val cancelButton = outlinedButton("발송 취소").apply {
            visibility = View.GONE
        }
        val openSmsAppButton = outlinedButton("문자 앱에서 직접 작성")
        content.addView(
            card {
                addView(statusText, matchWrap())
                addView(sendButton, matchWrap(topMargin = 16))
                addView(cancelButton, matchWrap(topMargin = 10))
                addView(openSmsAppButton, matchWrap(topMargin = 10))
            },
            matchWrap(topMargin = 18)
        )

        return EmergencyContactViews(
            root = root,
            backButton = backButton,
            contactSettingsCard = contactSettingsCard,
            contactNameInput = contactNameInput,
            contactPhoneInput = contactPhoneInput,
            pickContactButton = pickContactButton,
            saveButton = saveButton,
            deleteButton = deleteButton,
            statusText = statusText,
            sendButton = sendButton,
            cancelButton = cancelButton,
            openSmsAppButton = openSmsAppButton
        )
    }

    private fun card(contentBuilder: LinearLayout.() -> Unit): MaterialCardView {
        val cardContent = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(20), dp(20), dp(20))
            contentBuilder()
        }
        return MaterialCardView(this).apply {
            radius = dp(24).toFloat()
            cardElevation = 0f
            setCardBackgroundColor(ContextCompat.getColor(this@EmergencyContactActivity, R.color.via_surface))
            strokeColor = ContextCompat.getColor(this@EmergencyContactActivity, R.color.via_surface_outline)
            strokeWidth = dp(1)
            addView(cardContent, ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
        }
    }

    private fun titleText(text: String): TextView =
        TextView(this).apply {
            this.text = text
            setTextColor(ContextCompat.getColor(this@EmergencyContactActivity, R.color.via_on_surface))
            textSize = 34f
            setTypeface(typeface, android.graphics.Typeface.BOLD)
        }

    private fun bodyText(text: String): TextView =
        TextView(this).apply {
            this.text = text
            setTextColor(ContextCompat.getColor(this@EmergencyContactActivity, R.color.via_on_surface_variant))
            textSize = 16f
            setLineSpacing(dp(4).toFloat(), 1f)
        }

    private fun editText(hintText: String): EditText =
        EditText(this).apply {
            hint = hintText
            minHeight = dp(56)
            setTextColor(ContextCompat.getColor(this@EmergencyContactActivity, R.color.via_on_surface))
            setHintTextColor(ContextCompat.getColor(this@EmergencyContactActivity, R.color.via_on_surface_variant))
            textSize = 18f
        }

    private fun filledButton(text: String): MaterialButton =
        MaterialButton(this).apply {
            this.text = text
            isAllCaps = false
            textSize = 16f
            minHeight = dp(56)
            cornerRadius = dp(18)
            gravity = Gravity.CENTER
        }

    private fun outlinedButton(text: String): MaterialButton =
        MaterialButton(this, null, com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            this.text = text
            isAllCaps = false
            textSize = 16f
            minHeight = dp(56)
            cornerRadius = dp(18)
            gravity = Gravity.CENTER
        }

    private fun backButton(): MaterialButton =
        MaterialButton(this, null, com.google.android.material.R.attr.borderlessButtonStyle).apply {
            text = "뒤로"
            setIconResource(R.drawable.ic_arrow_back_24)
            iconGravity = MaterialButton.ICON_GRAVITY_TEXT_START
            iconPadding = dp(4)
            iconTint = android.content.res.ColorStateList.valueOf(
                ContextCompat.getColor(this@EmergencyContactActivity, R.color.via_on_surface)
            )
            isAllCaps = false
            textSize = 16f
            minHeight = dp(48)
            setTextColor(ContextCompat.getColor(this@EmergencyContactActivity, R.color.via_on_surface))
            gravity = Gravity.CENTER_VERTICAL or Gravity.START
            minWidth = 0
            setPadding(0, paddingTop, dp(12), paddingBottom)
            textAlignment = View.TEXT_ALIGNMENT_TEXT_START
        }

    private fun matchWrap(topMargin: Int = 0): LinearLayout.LayoutParams =
        LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply {
            if (topMargin > 0) {
                setMargins(0, dp(topMargin), 0, 0)
            }
        }

    private fun dp(value: Int): Int =
        (value * resources.displayMetrics.density).toInt()

    companion object {
        private const val AUTO_SEND_DELAY_MS = 5_000L
        const val EXTRA_AUTO_START_COUNTDOWN =
            "kr.co.gachon.pproject6.via.safety.AUTO_START_COUNTDOWN"
    }
}

private data class EmergencyContactViews(
    val root: View,
    val backButton: MaterialButton,
    val contactSettingsCard: View,
    val contactNameInput: EditText,
    val contactPhoneInput: EditText,
    val pickContactButton: MaterialButton,
    val saveButton: MaterialButton,
    val deleteButton: MaterialButton,
    val statusText: TextView,
    val sendButton: MaterialButton,
    val cancelButton: MaterialButton,
    val openSmsAppButton: MaterialButton
)
