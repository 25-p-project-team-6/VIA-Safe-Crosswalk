// Top-level build file where you can add configuration options common to all sub-projects/modules.
plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.kotlin.android) apply false
}

val externalBuildRoot =
    providers.gradleProperty("omxBuildRoot")
        .orElse(providers.environmentVariable("OMX_BUILD_ROOT"))
        .orElse(providers.provider { "${System.getProperty("java.io.tmpdir")}VIA-gradle-build" })

allprojects {
    layout.buildDirectory.set(
        java.io.File(externalBuildRoot.get(), project.name)
    )
}
