// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "JimssakCapture",
    platforms: [.iOS(.v16)],
    products: [.library(name: "JimssakCapture", targets: ["JimssakCapture"])],
    targets: [.target(name: "JimssakCapture")]
)
