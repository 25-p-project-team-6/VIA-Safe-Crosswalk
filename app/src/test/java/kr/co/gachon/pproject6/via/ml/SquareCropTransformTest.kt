package kr.co.gachon.pproject6.via.ml

import org.junit.Assert.assertEquals
import org.junit.Test

class SquareCropTransformTest {
    @Test
    fun portraitFrameCropsTopAndBottomThenMapsBackToSourceCoordinates() {
        val transform =
            SquareCropTransform.from(
                sourceWidth = 576,
                sourceHeight = 768,
                inputWidth = 512,
                inputHeight = 512
            )

        assertEquals(0, transform.cropLeft)
        assertEquals(96, transform.cropTop)
        assertEquals(576, transform.cropSize)

        val mapped =
            transform.inputBoundsToSourceNormalized(
                left = 0.25f,
                top = 0.25f,
                right = 0.75f,
                bottom = 0.75f
            )

        requireNotNull(mapped)
        assertEquals(0.25f, mapped.left, 0.0001f)
        assertEquals((96f + 144f) / 768f, mapped.top, 0.0001f)
        assertEquals(0.75f, mapped.right, 0.0001f)
        assertEquals((96f + 432f) / 768f, mapped.bottom, 0.0001f)
    }

    @Test
    fun landscapeFrameCropsLeftAndRightThenMapsBackToSourceCoordinates() {
        val transform =
            SquareCropTransform.from(
                sourceWidth = 768,
                sourceHeight = 576,
                inputWidth = 512,
                inputHeight = 512
            )

        assertEquals(96, transform.cropLeft)
        assertEquals(0, transform.cropTop)
        assertEquals(576, transform.cropSize)

        val mapped =
            transform.inputBoundsToSourceNormalized(
                left = 0.25f,
                top = 0.25f,
                right = 0.75f,
                bottom = 0.75f
            )

        requireNotNull(mapped)
        assertEquals((96f + 144f) / 768f, mapped.left, 0.0001f)
        assertEquals(0.25f, mapped.top, 0.0001f)
        assertEquals((96f + 432f) / 768f, mapped.right, 0.0001f)
        assertEquals(0.75f, mapped.bottom, 0.0001f)
    }

    @Test
    fun squareFrameKeepsWholeSourceCoordinates() {
        val transform =
            SquareCropTransform.from(
                sourceWidth = 512,
                sourceHeight = 512,
                inputWidth = 512,
                inputHeight = 512
            )

        assertEquals(0, transform.cropLeft)
        assertEquals(0, transform.cropTop)
        assertEquals(512, transform.cropSize)

        val mapped =
            transform.inputBoundsToSourceNormalized(
                left = 0.1f,
                top = 0.2f,
                right = 0.8f,
                bottom = 0.9f
            )

        requireNotNull(mapped)
        assertEquals(0.1f, mapped.left, 0.0001f)
        assertEquals(0.2f, mapped.top, 0.0001f)
        assertEquals(0.8f, mapped.right, 0.0001f)
        assertEquals(0.9f, mapped.bottom, 0.0001f)
    }
}
