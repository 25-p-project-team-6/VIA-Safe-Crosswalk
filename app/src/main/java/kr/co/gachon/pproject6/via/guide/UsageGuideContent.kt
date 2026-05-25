package kr.co.gachon.pproject6.via.guide

data class UsageGuideSection(
    val title: String,
    val body: String
)

object UsageGuideContent {
    const val screenTitle: String = "사용 안내"
    const val intro: String = "VIA는 보행 판단을 보조합니다. 실제 이동 전 주변을 직접 확인하세요."

    val feedbackOverview: UsageGuideSection =
        UsageGuideSection(
            title = "어떤 피드백을 받나요?",
            body = "VIA는 보행자 신호를 음성, 진동, 화면으로 알려줍니다. 아래 예시를 눌러 소리와 진동을 먼저 확인해 보세요."
        )

    val howItWorks: UsageGuideSection =
        UsageGuideSection(
            title = "어떻게 작동하나요?",
            body = "카메라로 보이는 보행자 신호를 확인하고, 현재 상태를 보조 안내로 전달합니다. 안내가 실제 상황과 다르면 주변 상황을 우선하세요."
        )

    val phoneHold: UsageGuideSection =
        UsageGuideSection(
            title = "휴대폰은 어떻게 들까요?",
            body = "휴대폰을 가슴 앞에 두고 카메라가 건너려는 방향을 보게 합니다. 신호가 보이지 않으면 몸을 천천히 돌려 방향을 다시 맞추세요."
        )

    val noFeedback: UsageGuideSection =
        UsageGuideSection(
            title = "피드백이 없으면?",
            body = "신호가 화면 밖에 있거나 사람·차량에 가려졌을 수 있습니다. 멈춘 뒤 휴대폰 방향을 다시 맞추고, 필요하면 주변 도움을 요청하세요."
        )

    val quickActions: UsageGuideSection =
        UsageGuideSection(
            title = "빠른 조작",
            body = "주변 횡단보도 안내와 블루투스 버튼 동작을 예시로 들어볼 수 있습니다. 예시는 실제 위치 조회나 비상 연락을 실행하지 않습니다."
        )

    val safetyNote: UsageGuideSection =
        UsageGuideSection(
            title = "안전 안내",
            body = "VIA는 보조 도구이며 보행 판단을 대신하지 않습니다. 건너기 전에는 항상 차량, 자전거, 주변 사람, 노면 상태를 직접 확인하세요."
        )

    val compactSections: List<UsageGuideSection> =
        listOf(howItWorks, phoneHold, noFeedback)

    val sections: List<UsageGuideSection> =
        compactSections + quickActions + safetyNote
}
