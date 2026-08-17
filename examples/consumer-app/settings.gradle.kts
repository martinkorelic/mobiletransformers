pluginManagement {
    repositories {
        gradlePluginPortal()
        google()
        mavenCentral()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        // mavenLocal() FIRST: this example exists to consume the artifact just published by
        // scripts/publish_local_maven.sh, so it must win over any remote of the same coordinates.
        mavenLocal()
        google()
        mavenCentral()
    }
}

rootProject.name = "mobiletransformers-consumer"
include(":app")
