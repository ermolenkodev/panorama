import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    kotlin("jvm") version "1.6.20"
}

group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.openpnp:opencv:4.5.1-2")
    implementation("org.jetbrains.kotlinx:kotlin-deeplearning-api:0.4.0")
    implementation("org.jetbrains.kotlinx:kotlin-deeplearning-onnx:0.4.0")
    implementation("org.jetbrains.kotlinx:kotlin-deeplearning-visualization:0.4.0")
    implementation("org.tensorflow:libtensorflow:1.15.0")
    implementation("org.tensorflow:libtensorflow_jni_gpu:1.15.0")
    implementation("org.jetbrains.kotlinx:multik-core:0.2.0-dev-3")
    implementation("org.jetbrains.kotlinx:multik-openblas:0.2.0-dev-3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.6.2")
    testImplementation(kotlin("test"))
}

tasks.test {
    useJUnitPlatform()
}

tasks.withType<KotlinCompile> {
    kotlinOptions.jvmTarget = "1.8"
}