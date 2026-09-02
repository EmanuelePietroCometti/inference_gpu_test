#pragma once
#include <windows.h>
#include <tchar.h>

//
// IPC_Shared.h
// Shared Memory-Mapped-File (MMF) contract between the external controller
// (producer / PLC) and the AI inference processes. This header is the SINGLE
// source of truth for the wire layout and MUST stay byte-identical on both
// sides. It is a superset of the ONNX_inference contract: the classic
// single-image fields are preserved unchanged (so the two executables remain
// interoperable on the same controller) and the batch extension is appended.
//
// Design note (inference_gpu specific):
//   inference_gpu runs a batch-17 pipeline. The input MMF carries kBatchSize
//   contiguous images per publish, organised as a ring of controlPoint.ringSlots
//   slots so the producer can stay ahead of the asynchronous pipeline without
//   overwriting a batch still in flight. The RING DEPTH is chosen by the
//   controller at configuration time (controlPoint.ringSlots) and the inference
//   program only reads it — it is NOT a compile-time constant. Per-image results
//   (anomaly score + OK/REJECT) travel in a dedicated batchResultBlock; the
//   per-image anomaly overlay travels in the result-image MMF.
//

// ============================================================================
// BATCH CONSTANTS
// ============================================================================

// Fixed inference batch size (requirement). One publish = kBatchSize images.
constexpr DWORD kBatchSize = 17;

// COMPILE-TIME CEILING only (like controlPointsList::points[1024]): the maximum
// ring depth the wire layout can ever hold. It sizes the fixed frameId[] array
// in the input header, nothing else. The ACTUAL ring depth is chosen at runtime
// by the CONTROLLER and passed in controlPoint.ringSlots (must be <= this); the
// inference program just reads it. Do NOT tune this per deployment — it is an
// ABI constant, so raising it is the only change that needs both sides rebuilt.
constexpr DWORD kMaxRingSlots = 32;

// ============================================================================
// ENUMS  (unchanged from the ONNX_inference contract)
// ============================================================================

// Manages the state of the inference result
enum class InferenceState : WORD {
    IDLE = 0,
    RESULT_READY = 1,
    PENDING = 2,
    ERROR_DETECTED = 4
};

// Manages the operational status of the control point
enum class PointState : WORD {
    IDLE = 0,
    CONFIGURED = 1,
    UPDATE_PENDING = 2,
    QUIT = 3,
    ERROR_DETECTED = 4
};

// Manages the global synchronization state
enum class ListState : WORD {
    IDLE = 0,
    CONFIGURED = 1,
    UPDATE_PENDING = 2,
    QUIT = 3,
    ERROR_DETECTED = 4
};

// Defines the type of AI task
enum class InferenceType : WORD {
    ANOMALY = 0,
    CLASSIFICATION = 1,
    OBJECT_DETECTION = 2
};

// Per-image verdict inside a batch result
enum class PatchStatus : WORD {
    OK = 0,
    REJECT = 1,
    ERROR_DETECTED = 0xFFFF
};

// ============================================================================
// STRUCTS  (classic single-image section, unchanged)
// ============================================================================

// Contains the output produced by the AI model (legacy single-image path).
// Kept for wire compatibility; batch results use batchResultBlock instead.
typedef struct resultInference {
    InferenceState state;
    DWORD sizeX;
    DWORD sizeY;
    TCHAR json[1024];
} resultInference, * PTresultInference;

// Defines the configuration parameters for a single control point.
// The batch fields are appended AT THE END so the offsets of every classic
// field stay identical to the ONNX_inference layout.
typedef struct controlPoint {
    DWORD idPunto;
    DWORD sizeX;            // width  of ONE input image (model image size)
    DWORD sizeY;            // height of ONE input image
    DWORD bpp;             // bits per pixel of ONE input image (e.g. 24)
	DWORD inferenceThreads;   // number of threads the inference program should run
    TCHAR pathModello[512];// ONNX model path; contract lives in its metadata
    TCHAR mutexName[128];
    TCHAR resultsEventName[128];
    TCHAR eventReadyName[128];
    PointState status;
    resultInference results;
    InferenceType inferenceType;

    // ---- batch extension (appended) ----
    DWORD batchSize;       // = kBatchSize for the inference_gpu provider
    DWORD loopBatch1;      // 0 = no, 1 = loop the first batch (for testing)
    DWORD ringSlots;       // ring depth / max in-flight, chosen by the CONTROLLER
                           // (1..kMaxRingSlots). The worker reads it, never sets it.
} controlPoint, * PTcontrolPoint;

// Manages the global list of control points for IPC (unchanged)
typedef struct controlPointsList {
    DWORD numPunti;
    ListState state;
    TCHAR listMutexName[128];
    TCHAR listEventTriggerName[128];
    TCHAR listEventAckName[128];
    controlPoint points[1024];
} controlPointsList, * PTcontrolPointsList;

// ============================================================================
// BATCH WIRE STRUCTS
// ============================================================================

// Header at offset 0 of the input MMF (MMF_{id}_IMAGE). The producer writes a
// batch into slot (publishSeq % ringSlots), records its frameId, then publishes
// 'publishSeq' (monotonic) and signals eventReady. The worker keeps a local
// consumed counter and drains every sequence up to publishSeq, so coalesced
// auto-reset events lose nothing. frameId[] is sized to the ABI ceiling; only
// the first controlPoint.ringSlots entries are used.
//
//   [ batchInputHeader | slot0 (kBatchSize imgs) | slot1 | ... ]
//

/// Page size for DMA alignment, Slotsw must start on a page boundary.
constexpr SIZE_T kPageSize = 4096; // round up to a page for alignment
typedef struct alignas(kPageSize) batchInputHeader {
    volatile LONG publishSeq;            // last published sequence (monotonic)
    DWORD         frameId[kMaxRingSlots];// frameId stored per ring slot (ceiling-sized)
    DWORD         _reserved;
    BYTE          _pad[kPageSize - (sizeof(LONG) + sizeof(DWORD) * (kMaxRingSlots + 1))];
} batchInputHeader;

// One per-image result inside a batch
typedef struct patchResult {
    float       anomalyScore;  // raw model score
    PatchStatus status;        // OK / REJECT / ERROR_DETECTED
    WORD        _pad;
} patchResult;

// Batch results block, written into one ring slot of MMF_{id}_RESULT by the
// worker. The pipeline runs several batches in flight, so results are a ring of
// controlPoint.ringSlots blocks (slot = seq % ringSlots). 'state' is written
// LAST (with a release barrier) so a reader that observes RESULT_READY sees a
// fully written block; 'seq' (monotonic batch index) + 'frameId' correlate it
// to the input and let the reader ignore a stale slot it already consumed.
typedef struct batchResultBlock {
    InferenceState state;               // RESULT_READY / ERROR_DETECTED (release flag)
    DWORD          seq;                 // monotonic batch index (ring correlation)
    DWORD          frameId;             // producer-assigned frame id
    DWORD          batchSize;           // = kBatchSize
    DWORD          mapH, mapW;          // model anomaly-map geometry (per image)
    DWORD          overlayBytesPerImage;// size of one overlay in the result-image MMF
    patchResult    results[kBatchSize];
} batchResultBlock;

// ============================================================================
// MMF SIZE HELPERS  (identical math on both sides; all driven by cp.ringSlots)
// ============================================================================

// Effective ring depth for a control point (defensive clamp to [1, ceiling]).
inline DWORD IpcRingSlots(const controlPoint& cp) {
    DWORD n = cp.ringSlots;
    if (n < 1) n = 1;
    if (n > kMaxRingSlots) n = kMaxRingSlots;
    return n;
}

// Bytes of ONE input/overlay image
inline SIZE_T IpcImageBytes(const controlPoint& cp) {
    return static_cast<SIZE_T>(cp.sizeX) * cp.sizeY * (cp.bpp / 8);
}

// Total bytes of the input MMF: header + ring of (ringSlots) batches
inline SIZE_T IpcInputMmfBytes(const controlPoint& cp) {
    return sizeof(batchInputHeader)
        + static_cast<SIZE_T>(IpcRingSlots(cp)) * cp.batchSize * IpcImageBytes(cp);
}

// Byte offset of ring slot 'slot' inside the input MMF
inline SIZE_T IpcInputSlotOffset(const controlPoint& cp, DWORD slot) {
    return sizeof(batchInputHeader)
        + static_cast<SIZE_T>(slot) * cp.batchSize * IpcImageBytes(cp);
}

// Bytes of ONE overlay batch (kBatchSize overlays)
inline SIZE_T IpcOverlayBatchBytes(const controlPoint& cp) {
    return static_cast<SIZE_T>(cp.batchSize) * IpcImageBytes(cp);
}

// Total bytes of the result-image MMF: a ring of (ringSlots) overlay batches
inline SIZE_T IpcResultImageMmfBytes(const controlPoint& cp) {
    return static_cast<SIZE_T>(IpcRingSlots(cp)) * IpcOverlayBatchBytes(cp);
}

// Byte offset of ring slot 'slot' inside the result-image MMF
inline SIZE_T IpcResultImageSlotOffset(const controlPoint& cp, DWORD slot) {
    return static_cast<SIZE_T>(slot) * IpcOverlayBatchBytes(cp);
}

// Total bytes of the result-block MMF: a ring of (ringSlots) blocks
inline SIZE_T IpcResultBlockMmfBytes(const controlPoint& cp) {
    return static_cast<SIZE_T>(IpcRingSlots(cp)) * sizeof(batchResultBlock);
}
