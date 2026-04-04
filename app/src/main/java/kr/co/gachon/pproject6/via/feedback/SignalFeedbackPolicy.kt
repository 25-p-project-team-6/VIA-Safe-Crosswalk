package kr.co.gachon.pproject6.via.feedback

class SignalFeedbackPolicy(
    private val timingConfig: SignalFeedbackTimingConfig = SignalFeedbackTimingConfig(),
    private val timeProvider: () -> Long = System::currentTimeMillis
) {
    private var activeSignature: FeedbackSignature? = null
    private var lastEmissionAt: Long = Long.MIN_VALUE

    fun shouldEmit(
        signatureKey: String,
        family: FeedbackRepeatFamily
    ): Boolean {
        val currentTime = timeProvider()
        val signature = FeedbackSignature(signatureKey, family)

        if (signature != activeSignature) {
            activeSignature = signature
            lastEmissionAt = currentTime
            return true
        }

        val repeatInterval =
            if (family == FeedbackRepeatFamily.WAIT_LIKE) {
                timingConfig.waitRepeatIntervalMs
            } else {
                timingConfig.actionRepeatIntervalMs
            }

        if (lastEmissionAt == Long.MIN_VALUE || currentTime - lastEmissionAt >= repeatInterval) {
            lastEmissionAt = currentTime
            return true
        }

        return false
    }

    fun clear() {
        activeSignature = null
        lastEmissionAt = Long.MIN_VALUE
    }
}

private data class FeedbackSignature(
    val key: String,
    val family: FeedbackRepeatFamily
)

enum class FeedbackRepeatFamily {
    ACTION_LIKE,
    WAIT_LIKE
}

data class SignalFeedbackTimingConfig(
    val actionRepeatIntervalMs: Long = 4_000L,
    val waitRepeatIntervalMs: Long = 8_000L
)
