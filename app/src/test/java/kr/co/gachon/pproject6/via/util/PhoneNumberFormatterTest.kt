package kr.co.gachon.pproject6.via.util

import org.junit.Assert.assertEquals
import org.junit.Test

class PhoneNumberFormatterTest {
    @Test
    fun normalizesContactPickerSpacingToDigitsOnly() {
        assertEquals("00000000000", PhoneNumberFormatter.normalizeForStorage("000 0 000 000-0"))
    }

    @Test
    fun formatsElevenDigitMobileStyleNumbers() {
        assertEquals("010-1234-5678", PhoneNumberFormatter.formatForDisplay("01012345678"))
        assertEquals("000-0000-0000", PhoneNumberFormatter.formatForDisplay("000 0 000 000-0"))
    }

    @Test
    fun formatsTenDigitAndSeoulNumbers() {
        assertEquals("031-123-4567", PhoneNumberFormatter.formatForDisplay("0311234567"))
        assertEquals("02-1234-5678", PhoneNumberFormatter.formatForDisplay("0212345678"))
        assertEquals("02-123-4567", PhoneNumberFormatter.formatForDisplay("021234567"))
    }

    @Test
    fun convertsKoreanInternationalMobileForStorageAndDisplay() {
        assertEquals("01012345678", PhoneNumberFormatter.normalizeForStorage("+82 10 1234 5678"))
        assertEquals("010-1234-5678", PhoneNumberFormatter.formatForDisplay("+82 10 1234 5678"))
    }
}
