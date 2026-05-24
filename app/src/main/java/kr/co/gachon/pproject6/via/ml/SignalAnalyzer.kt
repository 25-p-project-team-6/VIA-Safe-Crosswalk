package kr.co.gachon.pproject6.via.ml

import android.graphics.Bitmap
import kr.co.gachon.pproject6.via.context.CrossingSupportSnapshot
import kr.co.gachon.pproject6.via.ui.OverlayView
import java.util.ArrayDeque

class SignalAnalyzer(
    private val objectTracker: ObjectTracker = ObjectTracker(),
    private val walkSignalPolicy: ConservativeWalkSignalPolicy =
        ConservativeWalkSignalPolicy(GuidanceTuningDefaults.walkSignalConfig),
    private val occupancyEvaluator: CrosswalkOccupancyEvaluator =
        CrosswalkOccupancyEvaluator(GuidanceTuningDefaults.occupancyConfig),
    private val timeProvider: () -> Long = System::currentTimeMillis
) {
    private var hasSeenAnyTarget = false
    private var lastTargetVisible = false
    private var lastMatchedClusterId: String? = null
    private val recentTargetReacquireTimes = ArrayDeque<Long>()
    private val recentMatchedClusterChangeTimes = ArrayDeque<Long>()

    fun analyze(
        bitmap: Bitmap,
        rawBoxes: List<OverlayView.BoundingBox>,
        enableTrafficLogic: Boolean,
        enableHighlight: Boolean,
        crossingSupportSnapshot: CrossingSupportSnapshot = CrossingSupportSnapshot()
    ): SignalAnalysisResult {
        if (!enableTrafficLogic) {
            reset()
            return SignalAnalysisResult(
                boxesToShow = rawBoxes,
                targetBox = null,
                targetScore = 0f,
                targetClassName = "None",
                trafficLightCount = 0,
                vehicleTrafficLightCount = 0,
                multipleSignalDetected = false,
                needsZoomSuggestion = false,
                targetRecentlyReacquired = false,
                recentMatchedClusterChangeCount = 0,
                trafficState = TrafficLightState.UNKNOWN,
                userGuidanceState = UserGuidanceState.WAIT,
                guidancePhase = GuidancePhase.WAITING_FOR_RED_BASELINE,
                guidanceBlockReason = GuidanceBlockReason.NO_SIGNAL,
                guidanceContinuityTier = GuidanceContinuityTier.NONE,
                handoffDecision = CrosswalkHandoffDecision.NONE,
                crossingSupportSnapshot = crossingSupportSnapshot,
                occupancyCaution = false,
                occupancyCautionLabels = emptyList()
            )
        }

        val correctedBoxes = PostProcessor.applyColorCorrection(bitmap, rawBoxes)
        val targetData = objectTracker.selectTarget(correctedBoxes)

        val targetBox = targetData?.first
        if (targetBox != null && enableHighlight) {
            targetBox.isTarget = true
        }
        val now = timeProvider()
        val targetSignals = correctedBoxes.filter { DetectionLabels.isPedestrianSignal(it.clsName) }
        val vehicleSignals = correctedBoxes.filter { DetectionLabels.isVehicleSignal(it.clsName) }
        val targetArea = targetBox?.let { it.box.width() * it.box.height() } ?: 0f
        val targetRecentlyReacquired = updateTargetReacquire(targetBox != null, now)
        val recentMatchedClusterChangeCount =
            updateMatchedClusterChange(
                crossingSupportSnapshot.mapProximitySnapshot.matchedClusterId,
                now
            )
        val multipleSignalDetected = targetSignals.size >= 2 ||
            (targetSignals.isNotEmpty() && vehicleSignals.isNotEmpty())
        val needsZoomSuggestion =
            targetSignals.isNotEmpty() &&
                (multipleSignalDetected || targetArea < GuidanceTuningDefaults.advisoryConfig.smallTargetAreaThreshold)

        val trafficState = PostProcessor.updateTrafficLightState(targetBox)
        val guidanceDecision = walkSignalPolicy.update(
            state = trafficState,
            crossingSupportSnapshot = crossingSupportSnapshot
        )
        val activeOccupancy =
            occupancyEvaluator.findActiveOccupancy(
                boxes = correctedBoxes,
                eligible = guidanceDecision.state == UserGuidanceState.GO
            )

        return SignalAnalysisResult(
            boxesToShow = correctedBoxes,
            targetBox = targetBox,
            targetScore = targetData?.second ?: 0f,
            targetClassName = targetBox?.clsName ?: "None",
            trafficLightCount = targetSignals.size,
            vehicleTrafficLightCount = vehicleSignals.size,
            multipleSignalDetected = multipleSignalDetected,
            needsZoomSuggestion = needsZoomSuggestion,
            targetRecentlyReacquired = targetRecentlyReacquired,
            recentMatchedClusterChangeCount = recentMatchedClusterChangeCount,
            trafficState = trafficState,
            userGuidanceState = guidanceDecision.state,
            guidancePhase = guidanceDecision.phase,
            guidanceBlockReason = guidanceDecision.blockReason,
            guidanceContinuityTier = guidanceDecision.continuityTier,
            handoffDecision = guidanceDecision.handoffDecision,
            crossingSupportSnapshot = crossingSupportSnapshot,
            occupancyCaution = activeOccupancy.isNotEmpty(),
            occupancyCautionLabels = activeOccupancy.map { it.clsName }.distinct().sorted()
        )
    }

    fun reset() {
        objectTracker.reset()
        PostProcessor.resetState()
        walkSignalPolicy.reset()
        occupancyEvaluator.reset()
        hasSeenAnyTarget = false
        lastTargetVisible = false
        lastMatchedClusterId = null
        recentTargetReacquireTimes.clear()
        recentMatchedClusterChangeTimes.clear()
    }

    private fun updateTargetReacquire(
        targetVisible: Boolean,
        now: Long
    ): Boolean {
        if (hasSeenAnyTarget && !lastTargetVisible && targetVisible) {
            recentTargetReacquireTimes.addLast(now)
        }
        if (targetVisible) {
            hasSeenAnyTarget = true
        }
        pruneOlderThan(recentTargetReacquireTimes, now - TARGET_REACQUIRE_WINDOW_MS)
        lastTargetVisible = targetVisible
        return recentTargetReacquireTimes.isNotEmpty()
    }

    private fun updateMatchedClusterChange(
        matchedClusterId: String?,
        now: Long
    ): Int {
        if (lastMatchedClusterId != null && matchedClusterId != null && matchedClusterId != lastMatchedClusterId) {
            recentMatchedClusterChangeTimes.addLast(now)
        }
        lastMatchedClusterId = matchedClusterId
        pruneOlderThan(recentMatchedClusterChangeTimes, now - CLUSTER_CHANGE_WINDOW_MS)
        return recentMatchedClusterChangeTimes.size
    }

    private fun pruneOlderThan(
        deque: ArrayDeque<Long>,
        cutoff: Long
    ) {
        while (deque.isNotEmpty() && deque.first() < cutoff) {
            deque.removeFirst()
        }
    }

    private companion object {
        private const val TARGET_REACQUIRE_WINDOW_MS = 4_000L
        private const val CLUSTER_CHANGE_WINDOW_MS = 10_000L
    }
}
