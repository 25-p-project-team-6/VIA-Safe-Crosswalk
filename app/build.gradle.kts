import com.android.build.gradle.tasks.GenerateBuildConfig
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

val buildStamp: String =
    SimpleDateFormat("MM-dd HH:mm", Locale.US).format(Date())

android {
    namespace = "kr.co.gachon.pproject6.via"
    compileSdk = 35

    defaultConfig {
        applicationId = "kr.co.gachon.pproject6.via"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
        buildConfigField("String", "BUILD_STAMP", "\"$buildStamp\"")
        buildConfigField("String", "MAP_DATA_MANIFEST_URL", "\"\"")
        buildConfigField("long", "MAP_DATA_REFRESH_INTERVAL_MS", "86400000L")
        buildConfigField("String", "KINETIC_MAP_API_BASE_URL", "\"https://api.map.kinetic.moe\"")
        buildConfigField("String", "KINETIC_MAP_STYLE", "\"default\"")

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    buildFeatures {
        buildConfig = true
    }
    androidResources {
        noCompress += "tflite"
    }
}

dependencies {

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.activity)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.camera.core)
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.lifecycle)
    implementation(libs.androidx.camera.view)
    implementation(libs.litert)
    implementation(libs.litert.gpu)
    implementation(libs.litert.gpu.api)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}

tasks.withType<GenerateBuildConfig>().configureEach {
    doNotTrackState("Gradle output snapshotting is unstable for generated BuildConfig files in this workspace")
}
