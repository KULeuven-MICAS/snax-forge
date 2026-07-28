// SNAX-FORGE Chisel build.
//
// Versions are deliberately identical to snax_cluster/hw/chisel/build.sbt.
// The generated datapaths are meant to be dropped into a SNAX shell alongside
// the hand-written streamer and accelerators; a Chisel version skew between
// the two trees produces link-time and FIRRTL-dialect failures that are far
// more painful to diagnose than they are to avoid.

ThisBuild / scalaVersion := "2.13.14"
ThisBuild / version      := "0.1.0"
ThisBuild / organization := "be.kuleuven.esat.micas"

val chiselVersion     = "6.4.0"
val chiseltestVersion = "6.0.0"
val scalatestVersion  = "3.2.19"

lazy val root = (project in file("."))
  .settings(
    name := "snax-forge-hw",
    libraryDependencies ++= Seq(
      "org.chipsalliance" %% "chisel"     % chiselVersion,
      "edu.berkeley.cs"   %% "chiseltest" % chiseltestVersion % "test",
      "org.scalatest"     %% "scalatest"  % scalatestVersion  % "test"
    ),
    scalacOptions ++= Seq(
      "-language:reflectiveCalls",
      "-deprecation",
      "-feature",
      // -Xcheckinit turns "field read before its initialiser ran" from a
      // silent null/zero into an exception. Chisel's `val io = IO(...)`
      // idiom makes that class of bug easy to write and hard to see.
      "-Xcheckinit"
    ),
    addCompilerPlugin(
      "org.chipsalliance" % "chisel-plugin" % chiselVersion cross CrossVersion.full
    ),
    // The specs poke and peek shared DUT handles; running them concurrently
    // inside one JVM produces flaky, uninterpretable failures.
    Test / parallelExecution := false
  )
