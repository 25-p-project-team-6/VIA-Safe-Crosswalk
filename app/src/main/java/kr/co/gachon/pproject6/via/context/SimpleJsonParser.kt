package kr.co.gachon.pproject6.via.context

internal object SimpleJsonParser {
    fun parseObject(json: String): Map<String, Any?> {
        val parser = Parser(json)
        val value = parser.parseValue()
        parser.skipWhitespace()
        require(parser.isAtEnd()) { "Unexpected trailing JSON content" }
        @Suppress("UNCHECKED_CAST")
        return value as? Map<String, Any?> ?: error("Expected JSON object")
    }

    private class Parser(
        private val source: String
    ) {
        private var index: Int = 0

        fun parseValue(): Any? {
            skipWhitespace()
            check(!isAtEnd()) { "Unexpected end of JSON" }
            return when (val current = source[index]) {
                '{' -> parseObject()
                '[' -> parseArray()
                '"' -> parseString()
                't' -> parseLiteral("true", true)
                'f' -> parseLiteral("false", false)
                'n' -> parseLiteral("null", null)
                '-', in '0'..'9' -> parseNumber()
                else -> error("Unexpected JSON token '$current' at index $index")
            }
        }

        fun skipWhitespace() {
            while (!isAtEnd() && source[index].isWhitespace()) {
                index += 1
            }
        }

        fun isAtEnd(): Boolean = index >= source.length

        private fun parseObject(): Map<String, Any?> {
            expect('{')
            skipWhitespace()
            val result = linkedMapOf<String, Any?>()
            if (peek('}')) {
                index += 1
                return result
            }

            while (true) {
                skipWhitespace()
                val key = parseString()
                skipWhitespace()
                expect(':')
                val value = parseValue()
                result[key] = value
                skipWhitespace()
                when {
                    peek('}') -> {
                        index += 1
                        return result
                    }
                    peek(',') -> index += 1
                    else -> error("Expected ',' or '}' at index $index")
                }
            }
        }

        private fun parseArray(): List<Any?> {
            expect('[')
            skipWhitespace()
            val result = mutableListOf<Any?>()
            if (peek(']')) {
                index += 1
                return result
            }

            while (true) {
                result += parseValue()
                skipWhitespace()
                when {
                    peek(']') -> {
                        index += 1
                        return result
                    }
                    peek(',') -> index += 1
                    else -> error("Expected ',' or ']' at index $index")
                }
            }
        }

        private fun parseString(): String {
            expect('"')
            val result = StringBuilder()
            while (!isAtEnd()) {
                val current = source[index++]
                when (current) {
                    '"' -> return result.toString()
                    '\\' -> {
                        check(!isAtEnd()) { "Unexpected end of string escape" }
                        val escaped = source[index++]
                        result.append(
                            when (escaped) {
                                '"', '\\', '/' -> escaped
                                'b' -> '\b'
                                'f' -> '\u000C'
                                'n' -> '\n'
                                'r' -> '\r'
                                't' -> '\t'
                                'u' -> parseUnicodeEscape()
                                else -> error("Unsupported escape sequence '\\$escaped'")
                            }
                        )
                    }
                    else -> result.append(current)
                }
            }
            error("Unterminated JSON string")
        }

        private fun parseUnicodeEscape(): Char {
            check(index + 4 <= source.length) { "Invalid unicode escape at index $index" }
            val hex = source.substring(index, index + 4)
            index += 4
            return hex.toInt(16).toChar()
        }

        private fun parseNumber(): Double {
            val start = index
            if (source[index] == '-') {
                index += 1
            }
            while (!isAtEnd() && source[index].isDigit()) {
                index += 1
            }
            if (!isAtEnd() && source[index] == '.') {
                index += 1
                while (!isAtEnd() && source[index].isDigit()) {
                    index += 1
                }
            }
            if (!isAtEnd() && (source[index] == 'e' || source[index] == 'E')) {
                index += 1
                if (!isAtEnd() && (source[index] == '+' || source[index] == '-')) {
                    index += 1
                }
                while (!isAtEnd() && source[index].isDigit()) {
                    index += 1
                }
            }
            return source.substring(start, index).toDouble()
        }

        private fun parseLiteral(
            literal: String,
            value: Any?
        ): Any? {
            check(source.regionMatches(index, literal, 0, literal.length)) {
                "Expected '$literal' at index $index"
            }
            index += literal.length
            return value
        }

        private fun expect(expected: Char) {
            check(!isAtEnd() && source[index] == expected) {
                "Expected '$expected' at index $index"
            }
            index += 1
        }

        private fun peek(expected: Char): Boolean {
            return !isAtEnd() && source[index] == expected
        }
    }
}

@Suppress("UNCHECKED_CAST")
internal fun Any?.jsonObjectOrNull(): Map<String, Any?>? = this as? Map<String, Any?>

internal fun Any?.jsonArrayOrEmpty(): List<Any?> = this as? List<Any?> ?: emptyList()

internal fun Any?.jsonStringOrNull(): String? = this as? String

internal fun Any?.jsonStringOrDefault(defaultValue: String): String {
    return jsonStringOrNull()?.takeIf { it.isNotBlank() } ?: defaultValue
}

internal fun Any?.jsonDoubleOrNull(): Double? {
    return when (this) {
        is Number -> this.toDouble()
        is String -> this.toDoubleOrNull()
        else -> null
    }
}
