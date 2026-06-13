package kr.co.gachon.pproject6.via.input

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class RemoteButtonPressClassifierTest {
    @Test
    fun shortPressEmitsShortActionOnKeyUp() {
        val classifier = RemoteButtonPressClassifier()

        assertNull(classifier.onDown(eventTimeMs = 1_000L, repeatCount = 0))

        assertEquals(RemoteButtonAction.SHORT_PRESS, classifier.onUp(eventTimeMs = 1_200L))
    }

    @Test
    fun longPressOnKeyUpDoesNotAlsoEmitShortPress() {
        val classifier = RemoteButtonPressClassifier()

        assertNull(classifier.onDown(eventTimeMs = 1_000L, repeatCount = 0))

        assertEquals(RemoteButtonAction.LONG_PRESS, classifier.onUp(eventTimeMs = 1_900L))
        assertNull(classifier.onUp(eventTimeMs = 1_950L))
    }

    @Test
    fun repeatedKeyDownCanEmitLongPressBeforeKeyUp() {
        val classifier = RemoteButtonPressClassifier()

        assertNull(classifier.onDown(eventTimeMs = 1_000L, repeatCount = 0))
        assertEquals(
            RemoteButtonAction.LONG_PRESS,
            classifier.onDown(eventTimeMs = 1_850L, repeatCount = 1)
        )

        assertNull(classifier.onUp(eventTimeMs = 1_900L))
    }

    @Test
    fun cooldownSuppressesImmediateSecondAction() {
        val classifier = RemoteButtonPressClassifier()

        classifier.onDown(eventTimeMs = 1_000L, repeatCount = 0)
        assertEquals(RemoteButtonAction.SHORT_PRESS, classifier.onUp(eventTimeMs = 1_100L))
        classifier.onDown(eventTimeMs = 1_500L, repeatCount = 0)

        assertNull(classifier.onUp(eventTimeMs = 1_600L))
    }
}
