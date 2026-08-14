#!/usr/bin/env swift
// Standalone Metal Q30 executor. Target binding awaits Agni's published contract.
import Foundation
import Metal

let defaultSource = #"""
#include <metal_stdlib>
using namespace metal;
kernel void q30_decay(device int *cells [[buffer(0)]],
                      device atomic_uint *saturations [[buffer(1)]],
                      constant int &coeff [[buffer(2)]],
                      constant uint &count [[buffer(3)]],
                      uint tid [[thread_position_in_grid]]) {
  if (tid >= count) return;
  long q = ((long)cells[tid] * (long)coeff + (1L << 29)) >> 30;
  if (q < -2147483648L) { cells[tid] = -2147483648; atomic_fetch_add_explicit(&saturations[0], 1, memory_order_relaxed); }
  else if (q > 2147483647L) { cells[tid] = 2147483647; atomic_fetch_add_explicit(&saturations[0], 1, memory_order_relaxed); }
  else cells[tid] = (int)q;
}
"""#

struct Reader {
  var data: Data; var at = 0
  mutating func take(_ n: Int) -> Data { defer { at += n }; return data.subdata(in: at..<(at + n)) }
  mutating func u16() -> Int { Int(UInt16(littleEndian: take(2).withUnsafeBytes { $0.loadUnaligned(as: UInt16.self) })) }
  mutating func u32() -> UInt32 { UInt32(littleEndian: take(4).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }) }
  mutating func i32() -> Int32 { Int32(bitPattern: u32()) }
  mutating func i32s(_ n: Int) -> [Int32] { (0..<n).map { _ in i32() } }
}
struct Vector { let name: String; let coeff: Int32; let cells: [Int32]; let want: [Int32]; let saturations: UInt32 }
func vectors(_ path: String) throws -> [Vector] {
  var r = Reader(data: try Data(contentsOf: URL(fileURLWithPath: path)))
  guard String(data: r.take(10), encoding: .ascii) == "Q30METAL1\0" else { throw NSError(domain: "fixture", code: 1) }
  var out: [Vector] = []
  while true {
    let n = r.u16(); if n == 0 { return out }
    guard r.at + n + 8 <= r.data.count else { throw NSError(domain: "fixture", code: 2) }
    let name = String(data: r.take(n), encoding: .ascii)!
    let coeff = r.i32(); let count = Int(r.u32())
    guard r.at + count * 8 + 4 <= r.data.count else { throw NSError(domain: "fixture", code: 3) }
    let cells = r.i32s(count), want = r.i32s(count), sat = r.u32()
    out.append(Vector(name: name, coeff: coeff, cells: cells, want: want, saturations: sat))
  }
}
func family(_ d: MTLDevice) -> String {
  var out: [String] = []
  if d.supportsFamily(.apple1) { out.append("apple1") }; if d.supportsFamily(.apple2) { out.append("apple2") }
  if d.supportsFamily(.apple3) { out.append("apple3") }; if d.supportsFamily(.apple4) { out.append("apple4") }
  if d.supportsFamily(.apple5) { out.append("apple5") }; if d.supportsFamily(.apple6) { out.append("apple6") }
  if d.supportsFamily(.apple7) { out.append("apple7") }; if d.supportsFamily(.apple8) { out.append("apple8") }
  if d.supportsFamily(.apple9) { out.append("apple9") }
  return out.joined(separator: ",")
}
func die(_ s: String) -> Never { fputs("q30-metal-runner: RED: \(s)\n", stderr); exit(1) }
let args = CommandLine.arguments
func value(_ flag: String) -> String? { guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }; return args[i + 1] }
guard let fixture = value("--fixture") else { die("need --fixture") }
// TODO=CONTRACT_PENDING: Agni must supply target MSL path, entry name, and exact buffer ABI.
let entry = value("--entry") ?? "q30_decay"
let source: String
do { source = try value("--source").map { try String(contentsOfFile: $0, encoding: .utf8) } ?? defaultSource } catch { die("source: \(error)") }
guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { die("Metal unavailable; CPU fallback forbidden") }
print("metal.device.raw=\(device.name)")
print("metal.family.raw=\(family(device))")
print("metal.executor=Metal")
let library: MTLLibrary; let pipeline: MTLComputePipelineState
do { library = try device.makeLibrary(source: source, options: nil); guard let fn = library.makeFunction(name: entry) else { die("entry missing: \(entry)") }; pipeline = try device.makeComputePipelineState(function: fn) } catch { die("Metal compile/pipeline: \(error)") }
let vs: [Vector]; do { vs = try vectors(fixture) } catch { die("fixture decode: \(error)") }
var failures = 0
for v in vs {
  let bytes = max(v.cells.count, 1) * MemoryLayout<Int32>.stride
  guard let cell = device.makeBuffer(length: max(bytes, 4), options: .storageModeShared), let sat = device.makeBuffer(length: 4, options: .storageModeShared) else { die("buffer allocation") }
  if !v.cells.isEmpty { v.cells.withUnsafeBytes { cell.contents().copyMemory(from: $0.baseAddress!, byteCount: bytes) } }
  sat.contents().assumingMemoryBound(to: UInt32.self).pointee = 0
  guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else { die("command encoding") }
  encoder.setComputePipelineState(pipeline); encoder.setBuffer(cell, offset: 0, index: 0); encoder.setBuffer(sat, offset: 0, index: 1)
  var coeff = v.coeff, count = UInt32(v.cells.count); encoder.setBytes(&coeff, length: 4, index: 2); encoder.setBytes(&count, length: 4, index: 3)
  if count > 0 { let width = min(pipeline.maxTotalThreadsPerThreadgroup, Int(count)); encoder.dispatchThreads(MTLSize(width: Int(count), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)) }
  encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
  if command.status == .error { die("GPU command error: \(command.error?.localizedDescription ?? "unknown")") }
  let got: [Int32] = Array(UnsafeBufferPointer<Int32>(start: cell.contents().assumingMemoryBound(to: Int32.self), count: v.cells.count))
  let gotSat = sat.contents().assumingMemoryBound(to: UInt32.self).pointee
  if got != v.want || gotSat != v.saturations { failures += 1; print("case=\(v.name) result=RED got_sat=\(gotSat) want_sat=\(v.saturations)") } else { print("case=\(v.name) result=green") }
}
if failures != 0 { die("parity failures=\(failures)") }
print("q30-metal-runner=green vectors=\(vs.count)")
