package kr.co.gachon.pproject6.via.ml

import android.graphics.RectF
import kr.co.gachon.pproject6.via.ui.OverlayView
import org.junit.Assert.assertEquals
import org.junit.Test

class PostProcessorTest {
    @Test
    fun humanSignalsMapToPedestrianTrafficState() {
        assertEquals(
            TrafficLightState.GREEN,
            PostProcessor.observedTrafficLightState(box(DetectionLabels.HUMAN_GREEN))
        )
        assertEquals(
            TrafficLightState.RED,
            PostProcessor.observedTrafficLightState(box(DetectionLabels.HUMAN_RED))
        )
    }

    @Test
    fun vehicleSignalsDoNotMapToPedestrianTrafficState() {
        assertEquals(
            TrafficLightState.UNKNOWN,
            PostProcessor.observedTrafficLightState(box(DetectionLabels.VEHICLE_GREEN))
        )
        assertEquals(
            TrafficLightState.UNKNOWN,
            PostProcessor.observedTrafficLightState(box(DetectionLabels.VEHICLE_RED))
        )
    }

    private fun box(label: String): OverlayView.BoundingBox {
        return OverlayView.BoundingBox(
            RectF(0.3f, 0.2f, 0.6f, 0.5f),
            label,
            0.9f
        )
    }
}
