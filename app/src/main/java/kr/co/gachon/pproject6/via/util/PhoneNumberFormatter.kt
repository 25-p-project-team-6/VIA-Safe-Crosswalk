package kr.co.gachon.pproject6.via.util

object PhoneNumberFormatter {
    fun normalizeForStorage(raw: String?): String {
        val value = raw?.trim().orEmpty()
        if (value.isBlank()) return ""

        val digits = value.filter(Char::isDigit)
        if (digits.isBlank()) return ""

        return if (value.startsWith("+82") && digits.startsWith("82") && digits.length > 2) {
            "0" + digits.drop(2)
        } else {
            digits
        }
    }

    fun formatForDisplay(raw: String?): String {
        val digits = normalizeForStorage(raw)
        if (digits.isBlank()) return ""

        return when {
            digits.startsWith("02") -> formatSeoulNumber(digits)
            digits.length == 8 -> "${digits.take(4)}-${digits.drop(4)}"
            digits.length <= 3 -> digits
            digits.length <= 7 -> "${digits.take(3)}-${digits.drop(3)}"
            digits.length <= 10 -> "${digits.take(3)}-${digits.drop(3).dropLast(4)}-${digits.takeLast(4)}"
            digits.length == 11 -> "${digits.take(3)}-${digits.drop(3).take(4)}-${digits.takeLast(4)}"
            else -> digits
        }
    }

    private fun formatSeoulNumber(digits: String): String {
        return when {
            digits.length <= 2 -> digits
            digits.length <= 6 -> "${digits.take(2)}-${digits.drop(2)}"
            digits.length <= 9 -> "${digits.take(2)}-${digits.drop(2).dropLast(4)}-${digits.takeLast(4)}"
            digits.length == 10 -> "${digits.take(2)}-${digits.drop(2).take(4)}-${digits.takeLast(4)}"
            else -> digits
        }
    }
}
