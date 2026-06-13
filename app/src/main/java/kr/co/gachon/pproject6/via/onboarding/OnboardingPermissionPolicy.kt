package kr.co.gachon.pproject6.via.onboarding

import android.Manifest

object OnboardingPermissionPolicy {
    fun hasRequiredPermissions(
        hasCameraPermission: Boolean,
        hasLocationPermission: Boolean,
        hasSmsPermission: Boolean
    ): Boolean {
        return hasCameraPermission && hasLocationPermission && hasSmsPermission
    }

    fun missingPermissions(
        hasCameraPermission: Boolean,
        hasLocationPermission: Boolean,
        hasSmsPermission: Boolean
    ): List<String> {
        return buildList {
            if (!hasCameraPermission) {
                add(Manifest.permission.CAMERA)
            }
            if (!hasLocationPermission) {
                add(Manifest.permission.ACCESS_FINE_LOCATION)
                add(Manifest.permission.ACCESS_COARSE_LOCATION)
            }
            if (!hasSmsPermission) {
                add(Manifest.permission.SEND_SMS)
            }
        }
    }
}
