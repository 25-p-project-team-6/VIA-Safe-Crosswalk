// Top-level build file where you can add configuration options common to all sub-projects/modules.
plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.kotlin.android) apply false
}

val externalBuildRoot =
    providers.gradleProperty("omxBuildRoot")
        .orElse(providers.environmentVariable("OMX_BUILD_ROOT"))

allprojects {
    externalBuildRoot.orNull?.let { buildRoot ->
        layout.buildDirectory.set(
            java.io.File(buildRoot, project.name)
        )
    }
}
