package kr.co.gachon.pproject6.via.onboarding

import android.Manifest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class OnboardingPermissionPolicyTest {
    @Test
    fun allRequiredPermissionsIncludeCameraLocationAndSms() {
        val missing =
            OnboardingPermissionPolicy.missingPermissions(
                hasCameraPermission = false,
                hasLocationPermission = false,
                hasSmsPermission = false
            )

        assertEquals(
            listOf(
                Manifest.permission.CAMERA,
                Manifest.permission.ACCESS_FINE_LOCATION,
                Manifest.permission.ACCESS_COARSE_LOCATION,
                Manifest.permission.SEND_SMS
            ),
            missing
        )
    }

    @Test
    fun coarseOrFineLocationSatisfiedBySingleLocationFlag() {
        val missing =
            OnboardingPermissionPolicy.missingPermissions(
                hasCameraPermission = true,
                hasLocationPermission = true,
                hasSmsPermission = true
            )

        assertTrue(missing.isEmpty())
        assertTrue(
            OnboardingPermissionPolicy.hasRequiredPermissions(
                hasCameraPermission = true,
                hasLocationPermission = true,
                hasSmsPermission = true
            )
        )
    }

    @Test
    fun smsPermissionIsRequiredForOnboardingCompletion() {
        assertFalse(
            OnboardingPermissionPolicy.hasRequiredPermissions(
                hasCameraPermission = true,
                hasLocationPermission = true,
                hasSmsPermission = false
            )
        )

        assertEquals(
            listOf(Manifest.permission.SEND_SMS),
            OnboardingPermissionPolicy.missingPermissions(
                hasCameraPermission = true,
                hasLocationPermission = true,
                hasSmsPermission = false
            )
        )
    }
}
