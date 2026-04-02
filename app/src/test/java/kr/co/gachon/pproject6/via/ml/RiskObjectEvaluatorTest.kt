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
                label = "car",
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
                label = "car",
                score = 0.8f,
                centerX = 0.05f,
                bottom = 0.2f,
                area = 0.01f
            )
        )
    }
}
