package kr.co.gachon.pproject6.via.input

enum class RemoteButtonAction {
    SHORT_PRESS,
    LONG_PRESS
}

class RemoteButtonPressClassifier(
    private val longPressThresholdMs: Long = DEFAULT_LONG_PRESS_THRESHOLD_MS,
    private val actionCooldownMs: Long = DEFAULT_ACTION_COOLDOWN_MS
) {
    private var downStartedAtMs: Long? = null
    private var longPressEmittedForCurrentPress = false
    private var lastActionAtMs: Long = Long.MIN_VALUE

    fun onDown(eventTimeMs: Long, repeatCount: Int): RemoteButtonAction? {
        if (repeatCount == 0 || downStartedAtMs == null) {
            downStartedAtMs = eventTimeMs
            longPressEmittedForCurrentPress = false
            return null
        }

        val startedAt = downStartedAtMs ?: return null
        return if (!longPressEmittedForCurrentPress &&
            eventTimeMs - startedAt >= longPressThresholdMs
        ) {
            longPressEmittedForCurrentPress = true
            emitIfNotCoolingDown(RemoteButtonAction.LONG_PRESS, eventTimeMs)
        } else {
            null
        }
    }

    fun onUp(eventTimeMs: Long): RemoteButtonAction? {
        val startedAt = downStartedAtMs
        downStartedAtMs = null
        if (startedAt == null || longPressEmittedForCurrentPress) {
            longPressEmittedForCurrentPress = false
            return null
        }

        val action =
            if (eventTimeMs - startedAt >= longPressThresholdMs) {
                RemoteButtonAction.LONG_PRESS
            } else {
                RemoteButtonAction.SHORT_PRESS
            }
        longPressEmittedForCurrentPress = false
        return emitIfNotCoolingDown(action, eventTimeMs)
    }

    private fun emitIfNotCoolingDown(
        action: RemoteButtonAction,
        eventTimeMs: Long
    ): RemoteButtonAction? {
        if (lastActionAtMs != Long.MIN_VALUE &&
            eventTimeMs - lastActionAtMs < actionCooldownMs
        ) {
            return null
        }
        lastActionAtMs = eventTimeMs
        return action
    }

    companion object {
        const val DEFAULT_LONG_PRESS_THRESHOLD_MS = 800L
        const val DEFAULT_ACTION_COOLDOWN_MS = 1_200L
    }
}
