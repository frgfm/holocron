import CoreML
import Foundation
import Darwin

func residentBytes() -> UInt64 {
    var info = mach_task_basic_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info_data_t>.size / MemoryLayout<natural_t>.size)
    let status = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    precondition(status == KERN_SUCCESS)
    return info.resident_size
}

func peakBytes() -> Int {
    var usage = rusage()
    precondition(getrusage(RUSAGE_SELF, &usage) == 0)
    return usage.ru_maxrss
}

let args = CommandLine.arguments
let directory = URL(fileURLWithPath: args[1])
let mode = args[2]
let baseline = residentBytes()
let config = MLModelConfiguration()
config.computeUnits = mode == "cpu" ? .cpuOnly : .all
let model = try MLModel(contentsOf: directory.appendingPathComponent("model.mlmodelc"), configuration: config)
let inputs = try (0..<3).map { index -> MLDictionaryFeatureProvider in
    let tensor = try MLMultiArray(shape: [1, 3, 224, 224], dataType: .float32)
    let bytes = try Data(contentsOf: directory.appendingPathComponent("input\(index).bin"))
    precondition(bytes.count == tensor.count * MemoryLayout<Float>.size)
    bytes.withUnsafeBytes { source in
        tensor.dataPointer.copyMemory(from: source.baseAddress!, byteCount: bytes.count)
    }
    return try MLDictionaryFeatureProvider(dictionary: ["images": tensor])
}

func predict(_ index: Int) throws -> [Float] {
    try autoreleasepool {
        let features = try model.prediction(from: inputs[index])
        let logits = features.featureValue(for: "logits")!.multiArrayValue!
        precondition(logits.shape.map(\.intValue) == [1, 3] && logits.dataType == .float32)
        return (0..<3).map { logits[$0].floatValue }
    }
}

let outputs = try (0..<3).map { try predict($0) }
for index in 0..<30 { _ = try predict(index % 3) }
let warm = residentBytes()
var durations = [Double]()
for index in 0..<150 {
    let start = DispatchTime.now().uptimeNanoseconds
    _ = try predict(index % 3)
    durations.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000)
}
durations.sort()
let result: [String: Any] = [
    "backend": "coreml_\(mode)", "median_ms": durations[75], "p95_ms": durations[142],
    "baseline_rss_bytes": baseline, "warm_rss_bytes": warm, "peak_rss_bytes": peakBytes(),
    "outputs": outputs
]
let data = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
print(String(data: data, encoding: .utf8)!)
