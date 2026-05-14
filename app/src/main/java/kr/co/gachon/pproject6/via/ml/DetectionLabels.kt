package kr.co.gachon.pproject6.via.ml

object DetectionLabels {
    const val BICYCLE = "bicycle"
    const val MOTORCYCLE = "motorcycle"
    const val VEHICLE = "vehicle"
    const val HUMAN_GREEN = "human_green"
    const val HUMAN_RED = "human_red"
    const val VEHICLE_GREEN = "vehicle_green"
    const val VEHICLE_RED = "vehicle_red"

    val sevenClassLabels: List<String> =
        listOf(
            BICYCLE,
            MOTORCYCLE,
            VEHICLE,
            HUMAN_GREEN,
            HUMAN_RED,
            VEHICLE_GREEN,
            VEHICLE_RED
        )

    val pedestrianSignalLabels: Set<String> = setOf(HUMAN_GREEN, HUMAN_RED)
    val vehicleSignalLabels: Set<String> = setOf(VEHICLE_GREEN, VEHICLE_RED)
    val occupancyLabels: Set<String> = setOf(BICYCLE, MOTORCYCLE, VEHICLE)

    fun modelFilesForActiveSchema(modelFiles: List<String>): List<String> {
        val tfliteFiles = modelFiles
            .filter { it.endsWith(".tflite", ignoreCase = true) }
            .sorted()
        val sevenClassFiles = tfliteFiles.filter { it.contains("7cls", ignoreCase = true) }
        val yolo26nSevenClassFiles =
            sevenClassFiles.filter { it.contains("yolo26n", ignoreCase = true) }
        return yolo26nSevenClassFiles.ifEmpty { sevenClassFiles.ifEmpty { tfliteFiles } }
    }

    fun isPedestrianSignal(label: String): Boolean =
        label.lowercase() in pedestrianSignalLabels

    fun isVehicleSignal(label: String): Boolean =
        label.lowercase() in vehicleSignalLabels

    fun isSignalLike(label: String): Boolean =
        isPedestrianSignal(label) || isVehicleSignal(label)

    fun pedestrianTrafficState(label: String): TrafficLightState {
        return when (label.lowercase()) {
            HUMAN_GREEN -> TrafficLightState.GREEN
            HUMAN_RED -> TrafficLightState.RED
            else -> TrafficLightState.UNKNOWN
        }
    }

    fun isPedestrianGreen(label: String): Boolean =
        label.equals(HUMAN_GREEN, ignoreCase = true)

    fun swappedPedestrianSignalLabel(label: String): String? {
        return when (label.lowercase()) {
            HUMAN_GREEN -> HUMAN_RED
            HUMAN_RED -> HUMAN_GREEN
            else -> null
        }
    }
}
