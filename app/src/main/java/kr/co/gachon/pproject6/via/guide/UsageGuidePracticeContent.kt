package kr.co.gachon.pproject6.via.guide

import kr.co.gachon.pproject6.via.feedback.SignalFeedbackPattern

enum class PracticeFeedbackCategory {
    SIGNAL,
    CONTROL
}

data class PracticeFeedbackExample(
    val id: String,
    val category: PracticeFeedbackCategory,
    val title: String,
    val description: String,
    val speechText: String,
    val hapticPattern: SignalFeedbackPattern?,
    val simulationOnly: Boolean = true
)

object UsageGuidePracticeContent {
    val signalExamples: List<PracticeFeedbackExample> =
        listOf(
            PracticeFeedbackExample(
                id = "practice_signal_red",
                category = PracticeFeedbackCategory.SIGNAL,
                title = "빨간불 예시",
                description = "빨간 보행자 신호가 보일 때의 안내 예시입니다.",
                speechText = "연습 예시입니다. 빨간불로 보입니다. 실제 이동 전 주변을 확인하세요.",
                hapticPattern = SignalFeedbackPattern.RED_CONFIRMED
            ),
            PracticeFeedbackExample(
                id = "practice_signal_green",
                category = PracticeFeedbackCategory.SIGNAL,
                title = "초록불 예시",
                description = "초록 보행자 신호가 확인됐을 때의 안내 예시입니다.",
                speechText = "연습 예시입니다. 초록불로 보입니다. 실제 이동 전 주변을 확인하세요.",
                hapticPattern = SignalFeedbackPattern.GREEN_CONFIRMED
            ),
            PracticeFeedbackExample(
                id = "practice_signal_green_caution",
                category = PracticeFeedbackCategory.SIGNAL,
                title = "주의 필요 예시",
                description = "초록 신호처럼 보여도 차량이나 주변 상황을 더 조심해야 할 때의 예시입니다.",
                speechText = "연습 예시입니다. 초록불로 보여도 차량을 주의하세요.",
                hapticPattern = SignalFeedbackPattern.GREEN_WITH_CAUTION
            ),
            PracticeFeedbackExample(
                id = "practice_signal_uncertain",
                category = PracticeFeedbackCategory.SIGNAL,
                title = "확인 필요 예시",
                description = "신호가 불확실하거나 카메라 시야가 부족할 때의 안내 예시입니다.",
                speechText = "연습 예시입니다. 확인이 필요합니다. 멈추고 주변을 확인하세요.",
                hapticPattern = SignalFeedbackPattern.WAIT_OR_UNCERTAIN
            )
        )

    val controlExamples: List<PracticeFeedbackExample> =
        listOf(
            PracticeFeedbackExample(
                id = "practice_control_nearby_crosswalk",
                category = PracticeFeedbackCategory.CONTROL,
                title = "주변 횡단보도 안내",
                description = "메인 화면 버튼이나 블루투스 버튼을 짧게 눌렀을 때 실행되는 기능입니다.",
                speechText = "연습 예시입니다. 가장 가까운 횡단보도는 전방 약 18미터에 있습니다.",
                hapticPattern = null
            ),
            PracticeFeedbackExample(
                id = "practice_control_short_press",
                category = PracticeFeedbackCategory.CONTROL,
                title = "블루투스 짧게 누르기",
                description = "블루투스 버튼을 짧게 눌렀을 때의 동작입니다.",
                speechText = "연습 예시입니다. 주변 횡단보도 안내를 실행합니다. 가장 가까운 횡단보도는 전방 약 18미터에 있습니다.",
                hapticPattern = null
            ),
            PracticeFeedbackExample(
                id = "practice_control_long_press",
                category = PracticeFeedbackCategory.CONTROL,
                title = "블루투스 길게 누르기",
                description = "블루투스 버튼을 길게 눌렀을 때의 동작입니다.",
                speechText = "연습 예시입니다. 비상 문자 화면을 엽니다. 5초 안에 취소하지 않으면 등록된 비상 연락처로 문자를 보냅니다.",
                hapticPattern = null
            )
        )

    val allExamples: List<PracticeFeedbackExample> =
        signalExamples + controlExamples
}
