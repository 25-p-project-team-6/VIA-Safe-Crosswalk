package kr.co.gachon.pproject6.via.guide

import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import kr.co.gachon.pproject6.via.R

class UsageGuideActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(createContentView())
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
        content.addView(titleText("앱 사용 안내"), matchWrap(topMargin = 8))
        content.addView(
            bodyText("VIA는 보행 판단을 대신하지 않고, 신호·횡단보도·비상 연락을 보조적으로 안내합니다."),
            matchWrap(topMargin = 8)
        )

        UsageGuideContent.sections.forEach { section ->
            content.addView(
                card {
                    addView(sectionTitleText(section.title), matchWrap())
                    addView(bodyText(section.body), matchWrap(topMargin = 8))
                },
                matchWrap(topMargin = 16)
            )
        }
        return root
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
            setTypeface(typeface, android.graphics.Typeface.BOLD)
        }

    private fun sectionTitleText(text: String): TextView =
        TextView(this).apply {
            this.text = text
            setTextColor(ContextCompat.getColor(this@UsageGuideActivity, R.color.via_on_surface))
            textSize = 22f
            setTypeface(typeface, android.graphics.Typeface.BOLD)
        }

    private fun bodyText(text: String): TextView =
        TextView(this).apply {
            this.text = text
            setTextColor(ContextCompat.getColor(this@UsageGuideActivity, R.color.via_on_surface_variant))
            textSize = 16f
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
            textAlignment = android.view.View.TEXT_ALIGNMENT_TEXT_START
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
