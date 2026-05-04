package kr.co.gachon.pproject6.via.guide

data class UsageGuideSection(
    val title: String,
    val body: String
)

object UsageGuideContent {
    val sections: List<UsageGuideSection> =
        listOf(
            UsageGuideSection(
                title = "VIA가 하는 일",
                body = "VIA는 보행자 신호와 주변 횡단보도 정보를 보조적으로 안내합니다. 실제 이동 전에는 반드시 차량, 자전거, 주변 사람, 노면 상태를 직접 확인해야 합니다."
            ),
            UsageGuideSection(
                title = "중요 안전 안내",
                body = "VIA는 보행 판단을 보조하는 앱이며 최종 판단을 대신하지 않습니다. 안내가 불확실하거나 주변 상황과 다르면 멈추고 주변 도움을 요청해 주세요."
            ),
            UsageGuideSection(
                title = "권한을 쓰는 이유",
                body = "카메라는 보행자 신호를 확인하는 데 사용합니다. 위치는 가까운 횡단보도 거리와 방향을 안내하는 데 사용합니다. SMS 권한은 사용자가 등록한 보호자나 기관에 비상 문자를 자동 발송할 때만 사용합니다."
            ),
            UsageGuideSection(
                title = "주요 기능",
                body = "신호 상태 안내는 카메라로 보이는 보행자 신호를 음성과 화면으로 알려줍니다. 주변 횡단보도 안내는 현재 위치 기준 가까운 횡단보도의 거리와 방향만 알려줍니다. 비상 문자는 5초 유예 후 등록된 연락처로 발송되며 취소할 수 있습니다."
            ),
            UsageGuideSection(
                title = "블루투스 버튼",
                body = "Space 키로 동작하는 블루투스 버튼을 사용할 수 있습니다. 짧게 누르면 주변 횡단보도 안내를 실행하고, 길게 누르면 비상 문자 5초 유예 화면을 엽니다. 화면 버튼으로도 같은 기능을 사용할 수 있습니다."
            ),
            UsageGuideSection(
                title = "상태 문구의 의미",
                body = "초록으로 보임은 카메라에서 보행자 초록 신호가 확인된 상태입니다. 빨간색으로 보임은 보행자 빨간 신호가 확인된 상태입니다. 신호 확인 불확실은 카메라 시야나 인식 신뢰도가 부족한 상태입니다. 다음 신호 대기 권장은 새로운 신호 주기를 기다리는 것이 더 안전하다는 뜻입니다."
            ),
            UsageGuideSection(
                title = "초기 최적화",
                body = "첫 실행 시 앱은 기기에서 사용할 AI 모델과 실행 방식을 짧게 측정합니다. 측정 중에는 휴대폰을 안정적으로 들고 카메라 권한을 유지해 주세요. 설정의 디버그 패널에서 모델과 FPS를 다시 확인할 수 있습니다."
            )
        )
}
