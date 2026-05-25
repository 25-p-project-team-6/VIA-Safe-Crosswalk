package kr.co.gachon.pproject6.via.guide

import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import kr.co.gachon.pproject6.via.R

class UsageGuideActivity : AppCompatActivity() {
    private lateinit var practiceFeedbackPlayer: PracticeFeedbackPlayer

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        practiceFeedbackPlayer = PracticeFeedbackPlayer(this)
        setContentView(createContentView())
    }

    override fun onPause() {
        super.onPause()
        practiceFeedbackPlayer.stop()
    }

    override fun onDestroy() {
        practiceFeedbackPlayer.release()
        super.onDestroy()
    }

    private fun createContentView(): ScrollView {
        val root = ScrollView(this).apply {
            setBackgroundColor(ContextCompat.getColor(this@UsageGuideActivity, R.color.via_background))
            isFillViewport = true
        }
        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(28), dp(20), dp(32))
        }
        root.addView(content, ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)

        content.addView(backButton().apply { setOnClickListener { finish() } })
        content.addView(titleText(UsageGuideContent.screenTitle), matchWrap(topMargin = 8))
        content.addView(bodyText(UsageGuideContent.intro), matchWrap(topMargin = 8))
        content.addView(feedbackPracticeCard(), matchWrap(topMargin = 20))

        UsageGuideContent.compactSections.forEach { section ->
            content.addView(infoCard(section), matchWrap(topMargin = 18))
        }
        content.addView(quickActionsCard(), matchWrap(topMargin = 18))
        content.addView(infoCard(UsageGuideContent.safetyNote), matchWrap(topMargin = 18))
        return root
    }

    private fun feedbackPracticeCard(): MaterialCardView =
        card {
            addView(sectionTitleText(UsageGuideContent.feedbackOverview.title), matchWrap())
            addView(bodyText(UsageGuideContent.feedbackOverview.body), matchWrap(topMargin = 8))
            addView(signalPracticeGrid(), matchWrap(topMargin = 20))
            addView(
                accentText("소리와 진동은 설정에 따라 다르게 느껴질 수 있습니다."),
                matchWrap(topMargin = 18)
            )
        }

    private fun signalPracticeGrid(): LinearLayout =
        LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            UsageGuidePracticeContent.signalExamples.chunked(2).forEachIndexed { rowIndex, rowExamples ->
                addView(
                    LinearLayout(this@UsageGuideActivity).apply {
                        orientation = LinearLayout.HORIZONTAL
                        rowExamples.forEachIndexed { index, example ->
                            addView(
                                signalPracticeItem(example),
                                LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply {
                                    if (index > 0) setMargins(dp(10), 0, 0, 0)
                                }
                            )
                        }
                    },
                    matchWrap(topMargin = if (rowIndex == 0) 0 else 16)
                )
            }
        }

    private fun signalPracticeItem(example: PracticeFeedbackExample): LinearLayout =
        LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            isClickable = true
            isFocusable = true
            minimumHeight = dp(156)
            contentDescription = "${example.title}. 두 번 탭하면 예시를 재생합니다."
            setOnClickListener { playPractice(example) }
            addView(
                signalCircle(example),
                LinearLayout.LayoutParams(dp(72), dp(72)).apply { gravity = Gravity.CENTER_HORIZONTAL }
            )
            addView(
                bodyText(example.title).apply {
                    gravity = Gravity.CENTER
                    textAlignment = View.TEXT_ALIGNMENT_CENTER
                    setTypeface(typeface, Typeface.BOLD)
                },
                matchWrap(topMargin = 8)
            )
            addView(playButton(example), matchWrap(topMargin = 8))
        }

    private fun signalCircle(example: PracticeFeedbackExample): TextView =
        TextView(this).apply {
            text = signalSymbol(example.id)
            gravity = Gravity.CENTER
            textSize = 30f
            setTextColor(Color.WHITE)
            background = GradientDrawable().apply {
                shape = GradientDrawable.OVAL
                setColor(signalColor(example.id))
            }
            contentDescription = example.title
        }

    private fun playButton(example: PracticeFeedbackExample): MaterialButton =
        MaterialButton(this, null, com.google.android.material.R.attr.borderlessButtonStyle).apply {
            text = "▶ 재생"
            isAllCaps = false
            minHeight = dp(44)
            setTextColor(ContextCompat.getColor(this@UsageGuideActivity, R.color.via_status_caution))
            setOnClickListener { playPractice(example) }
        }

    private fun quickActionsCard(): MaterialCardView =
        card {
            addView(sectionTitleText(UsageGuideContent.quickActions.title), matchWrap())
            addView(bodyText(UsageGuideContent.quickActions.body), matchWrap(topMargin = 8))
            UsageGuidePracticeContent.controlExamples.forEach { example ->
                addView(quickActionItem(example), matchWrap(topMargin = 12))
            }
        }

    private fun quickActionItem(example: PracticeFeedbackExample): MaterialCardView {
        val itemContent =
            LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(dp(18), dp(16), dp(18), dp(16))
                addView(
                    TextView(this@UsageGuideActivity).apply {
                        text = "▶ ${example.title}"
                        setTextColor(ContextCompat.getColor(this@UsageGuideActivity, R.color.via_on_surface))
                        textSize = 20f
                        setTypeface(typeface, Typeface.BOLD)
                    },
                    matchWrap()
                )
                addView(bodyText(example.description), matchWrap(topMargin = 6))
                addView(
                    accentText("두 번 탭하면 실제 동작 안내 예시를 재생합니다."),
                    matchWrap(topMargin = 8)
                )
            }

        return MaterialCardView(this).apply {
            radius = dp(18).toFloat()
            cardElevation = 0f
            setCardBackgroundColor(ContextCompat.getColor(this@UsageGuideActivity, R.color.via_accent_container))
            strokeColor = ContextCompat.getColor(this@UsageGuideActivity, R.color.via_surface_outline)
            strokeWidth = dp(1)
            isClickable = true
            isFocusable = true
            minimumHeight = dp(116)
            contentDescription = "${example.title}. ${example.description}. 두 번 탭하면 실제 동작 안내 예시를 재생합니다."
            setOnClickListener { playPractice(example) }
            addView(itemContent, ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
        }
    }

    private fun infoCard(section: UsageGuideSection): MaterialCardView =
        card {
            addView(sectionTitleText(section.title), matchWrap())
            addView(bodyText(section.body), matchWrap(topMargin = 8))
        }

    private fun playPractice(example: PracticeFeedbackExample) {
        val played = practiceFeedbackPlayer.play(example)
        if (!played) {
            Toast.makeText(this, "음성 엔진을 준비 중입니다. 잠시 후 다시 눌러 주세요.", Toast.LENGTH_SHORT).show()
        }
    }

    private fun signalSymbol(exampleId: String): String =
        when (exampleId) {
            "practice_signal_red" -> "✋"
            "practice_signal_green" -> "🚶"
            "practice_signal_green_caution" -> "!"
            else -> "?"
        }

    private fun signalColor(exampleId: String): Int =
        ContextCompat.getColor(
            this,
            when (exampleId) {
                "practice_signal_red" -> R.color.via_status_stop
                "practice_signal_green" -> R.color.via_status_go
                "practice_signal_green_caution" -> R.color.via_status_caution
                else -> R.color.via_status_wait
            }
        )

    private fun card(contentBuilder: LinearLayout.() -> Unit): MaterialCardView {
        val cardContent = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(20), dp(20), dp(20))
            contentBuilder()
        }
        return MaterialCardView(this).apply {
            radius = dp(24).toFloat()
            cardElevation = 0f
            setCardBackgroundColor(ContextCompat.getColor(this@UsageGuideActivity, R.color.via_surface))
            strokeColor = ContextCompat.getColor(this@UsageGuideActivity, R.color.via_surface_outline)
            strokeWidth = dp(1)
            addView(cardContent, ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
        }
    }

    private fun titleText(text: String): TextView =
        TextView(this).apply {
            this.text = text
            setTextColor(ContextCompat.getColor(this@UsageGuideActivity, R.color.via_on_surface))
            textSize = 34f
            setTypeface(typeface, Typeface.BOLD)
        }

    private fun sectionTitleText(text: String): TextView =
        TextView(this).apply {
            this.text = text
            setTextColor(ContextCompat.getColor(this@UsageGuideActivity, R.color.via_on_surface))
            textSize = 24f
            setTypeface(typeface, Typeface.BOLD)
        }

    private fun bodyText(text: String): TextView =
        TextView(this).apply {
            this.text = text
            setTextColor(ContextCompat.getColor(this@UsageGuideActivity, R.color.via_on_surface_variant))
            textSize = 16f
            setLineSpacing(dp(4).toFloat(), 1f)
        }

    private fun accentText(text: String): TextView =
        TextView(this).apply {
            this.text = text
            setTextColor(ContextCompat.getColor(this@UsageGuideActivity, R.color.via_status_caution))
            textSize = 15f
            setLineSpacing(dp(4).toFloat(), 1f)
        }

    private fun backButton(): MaterialButton =
        MaterialButton(this, null, com.google.android.material.R.attr.borderlessButtonStyle).apply {
            text = "뒤로"
            setIconResource(R.drawable.ic_arrow_back_24)
            iconGravity = MaterialButton.ICON_GRAVITY_TEXT_START
            iconPadding = dp(4)
            iconTint = android.content.res.ColorStateList.valueOf(
                ContextCompat.getColor(this@UsageGuideActivity, R.color.via_on_surface)
            )
            isAllCaps = false
            gravity = Gravity.CENTER_VERTICAL or Gravity.START
            minWidth = 0
            setPadding(0, paddingTop, dp(12), paddingBottom)
            textAlignment = View.TEXT_ALIGNMENT_TEXT_START
            textSize = 16f
            minHeight = dp(48)
            setTextColor(ContextCompat.getColor(this@UsageGuideActivity, R.color.via_on_surface))
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
}
