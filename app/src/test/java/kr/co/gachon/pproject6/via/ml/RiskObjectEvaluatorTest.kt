package kr.co.gachon.pproject6.via.ml

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class RiskObjectEvaluatorTest {
    private val evaluator = CrosswalkOccupancyEvaluator()

    @Test
    fun centeredLargeVehicleBecomesOccupancyCandidate() {
        assertTrue(
            evaluator.isOccupancyCandidate(
                label = DetectionLabels.VEHICLE,
                score = 0.8f,
                centerX = 0.5f,
                bottom = 0.8f,
                area = 0.12f
            )
        )
    }

    @Test
    fun smallOffCenterVehicleIsIgnored() {
        assertFalse(
            evaluator.isOccupancyCandidate(
                label = DetectionLabels.VEHICLE,
                score = 0.8f,
                centerX = 0.05f,
                bottom = 0.2f,
                area = 0.01f
            )
        )
    }

    @Test
    fun bicycleAndMotorcycleRemainOccupancyCandidates() {
        assertTrue(
            evaluator.isOccupancyCandidate(
                label = DetectionLabels.BICYCLE,
                score = 0.8f,
                centerX = 0.5f,
                bottom = 0.8f,
                area = 0.12f
            )
        )
        assertTrue(
            evaluator.isOccupancyCandidate(
                label = DetectionLabels.MOTORCYCLE,
                score = 0.8f,
                centerX = 0.5f,
                bottom = 0.8f,
                area = 0.12f
            )
        )
    }
}
