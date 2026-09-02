// inference_gpu_test.cpp
// Stand-in controller / PLC for the inference_gpu manager, driven by keyboard
// commands (same UX as ConsoleApplicationTestAI):
//
//     c : configure  -> writes one ANOMALY control point into the shared list,
//                       triggers the manager and waits (NO fixed timeout, so the
//                       first-time TensorRT engine build can take minutes) for
//                       the CONFIGURED ack; press 'q' during the wait to abort.
//     s : start      -> starts the pipelined inference (up to the in-flight
//                       batches in flight) iterating over the images folder.
//     x : stop       -> stops the inference loop and prints a run summary.
//     q : quit       -> stops inference (if running), sends QUIT, exits.
//
// Global IPC objects are acquired with create-or-open, so the provider and this
// controller can be launched in EITHER order. Each result batch is correlated to
// its input by the monotonic 'seq' (ring slot). Pipelined: while one batch runs
// on the GPU the next is preprocessed and the previous is postprocessed.
//
// Usage: inference-gpu-test <onnx_model_path> <images_dir> [W=256] [H=256]

#include "IPC_Shared.h"

#include <windows.h>
#include <conio.h>
#include <opencv2/opencv.hpp>
#include <fmt/core.h>
#include <fmt/xchar.h>

#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <semaphore>
#include <memory>
#include <cstring>
#include <cctype>
#include <filesystem>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Configuration / geometry
// ---------------------------------------------------------------------------
static std::string g_modelPath, g_imagesDir;
static int   g_W = 256, g_H = 256, g_inferenceThreads = 1;
static int   g_bpp = 24, g_channels = 3, g_B = (int)kBatchSize;
static int  g_loop1 = 0;
static size_t g_imgBytes = 0;
static std::vector<cv::Mat> g_images;

// ---------------------------------------------------------------------------
// IPC handles (global list + per-point + per-point MMFs)
// ---------------------------------------------------------------------------
static const DWORD    kId = 0;
static const wchar_t* kMutexName = L"CP_0_MUTEX";
static const wchar_t* kReadyName = L"CP_0_READY";
static const wchar_t* kResultsName = L"CP_0_RESULTS";

static HANDLE hList = NULL, hListMutex = NULL, hTrigger = NULL, hAck = NULL;
static PTcontrolPointsList pList = nullptr;
static controlPoint* cpPtr = nullptr;

static HANDLE hCpMutex = NULL, hCpReady = NULL, hCpResults = NULL;
static HANDLE hIn = NULL, hResImg = NULL, hRes = NULL;
static unsigned char* pIn = nullptr;
static unsigned char* pResImg = nullptr;
static batchResultBlock* pRes = nullptr;
static batchInputHeader* header = nullptr;

// ---------------------------------------------------------------------------
// Pipeline state
// ---------------------------------------------------------------------------
// In-flight budget / ring depth: how many batches may be outstanding at once.
// This is what the CONTROLLER imposes on the inference program (written into
// controlPoint.ringSlots at configuration). Chosen at runtime, <= kMaxRingSlots
// (the ABI ceiling). The semaphore's template max is that ceiling; it is
// constructed with the runtime budget in main().
static int g_inFlight = 5;
static std::unique_ptr<std::counting_semaphore<kMaxRingSlots>> g_permits;
static std::atomic<bool> g_stop{ false }, g_infRunning{ false }, g_configured{ false };
static std::atomic<bool> g_senderDone{ false }, g_savedFrame0{ false };
static std::atomic<long> g_sent{ 0 }, g_received{ 0 }, g_ok{ 0 }, g_reject{ 0 };
static std::atomic<long> g_drops{ 0 };          // ticks that couldn't push (pipeline busy)
static int g_frameIntervalMs = 40;              // fixed frame-grabber interval

// Warmup: the first g_warmupBatches results are NOT counted (GPU ramps to its
// steady clock and the pipeline fills). When that many results have arrived, the
// receiver captures baselines and restarts the measurement window from zero.
static int  g_warmupBatches = 100;
static std::atomic<bool> g_warming{ false };
static long g_baseReceived = 0, g_baseSent = 0, g_baseOk = 0, g_baseReject = 0, g_baseDrops = 0;
static size_t g_imgCursor = 0;
static int g_lastSeq[kMaxRingSlots];
static std::thread g_senderThread, g_receiverThread;
static std::chrono::high_resolution_clock::time_point g_tStart;
static std::atomic<bool> g_quitRequested{ false };

// End-to-end latency accounting: the sender stamps the publish time per ring
// slot (right before the MMF copy); the receiver measures now - that stamp when
// the matching result arrives. In-flight is bounded to g_inFlight, so a
// slot's stamp is never overwritten before its result is consumed. Written by
// the receiver thread only; read by the main thread after join.
static std::chrono::high_resolution_clock::time_point g_pubTime[kMaxRingSlots];
static double g_latSumMs = 0.0, g_latMinMs = 0.0, g_latMaxMs = 0.0;
static long   g_latCount = 0;
static constexpr double kTargetMsPerImage = 2.16; // GPU speed target for skrd4ad

namespace {

BOOL WINAPI CtrlHandler(DWORD type) {
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT || type == CTRL_CLOSE_EVENT) {
        g_quitRequested.store(true);
        return TRUE;
    }
    return FALSE;
}

template <typename OpenFn>
HANDLE OpenWithRetry(OpenFn fn, const char* what, int retries = 400, int sleepMs = 50) {
    for (int i = 0; i < retries; ++i) {
        HANDLE h = fn();
        if (h) return h;
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
    }
    fmt::print(stderr, "Timed out opening {}\n", what);
    return nullptr;
}

bool IsImageFile(const fs::path& p) {
    std::string ext = p.extension().string();
    for (auto& c : ext) c = (char)tolower((unsigned char)c);
    return ext == ".bmp" || ext == ".png" || ext == ".jpg" || ext == ".jpeg" ||
           ext == ".tif" || ext == ".tiff" || ext == ".webp";
}

} // namespace

// ---------------------------------------------------------------------------
// Pipeline threads
// ---------------------------------------------------------------------------
static void SenderLoop()
{
    long produced = 0;

    // Copies the next kBatchSize images into the current ring slot and triggers
    // the worker. Stamps the publish time for the end-to-end latency measurement.
    auto doPublish = [&]() {
        const LONG k = header->publishSeq;
        const DWORD slot = (DWORD)(k % (LONG)g_inFlight);
        g_pubTime[slot] = std::chrono::high_resolution_clock::now(); // t0: start of the copy
        unsigned char* slotPtr = pIn + IpcInputSlotOffset(*cpPtr, slot);
        for (int i = 0; i < g_B; ++i) {
            const cv::Mat& im = g_images[g_imgCursor++ % g_images.size()];
            std::memcpy(slotPtr + (size_t)i * g_imgBytes, im.data, g_imgBytes);
        }
        header->frameId[slot] = (DWORD)produced;
        MemoryBarrier();
        header->publishSeq = k + 1;
        SetEvent(hCpReady);
        ++produced;
        g_sent.store(produced);
    };

    if (g_frameIntervalMs <= 0) {
        // FREE-RUNNING: push as fast as the pipeline drains -> measures the
        // maximum sustainable throughput (the GPU speed limit). No frame drops.
        while (!g_stop.load()) {
            if (!g_permits->try_acquire_for(std::chrono::milliseconds(100))) continue;
            doPublish();
        }
    }
    else {
        // FIXED-RATE frame grabber: attempt one batch every g_frameIntervalMs. If
        // no ring slot is free (pipeline still busy), the frame is DROPPED and
        // counted instead of blocking — exactly like a camera that keeps
        // triggering regardless of whether the consumer kept up.
        const auto interval = std::chrono::milliseconds(g_frameIntervalMs);
        auto next = std::chrono::steady_clock::now();
        while (!g_stop.load()) {
            next += interval;
            std::this_thread::sleep_until(next);
            if (g_stop.load()) break;
            // Fell behind by >= one interval: skip missed ticks, don't burst.
            const auto now = std::chrono::steady_clock::now();
            while (next + interval <= now) next += interval;

            if (g_permits->try_acquire()) {
                doPublish();
            }
            else {
                long d = g_drops.fetch_add(1) + 1;
                fmt::print(stderr, "[TEST] ### FRAME DROP: pipeline busy at {} ms tick (total drops: {})\n",
                    g_frameIntervalMs, d);
                fflush(stderr);
            }
        }
    }
    g_senderDone.store(true);
}

static void ReceiverLoop()
{
    const auto* resBase = reinterpret_cast<const unsigned char*>(pRes);
    bool draining = false;
    std::chrono::steady_clock::time_point drainDeadline{};

    while (true) {
        WaitForSingleObject(hCpResults, 200);
        for (int s = 0; s < g_inFlight; ++s) {
            auto* rb = reinterpret_cast<const batchResultBlock*>(resBase + (size_t)s * sizeof(batchResultBlock));
            if (rb->state != InferenceState::RESULT_READY) continue;
            MemoryBarrier();
            const int seq = (int)rb->seq;
            if (seq == g_lastSeq[s]) continue;
            g_lastSeq[s] = seq;

            int ok = 0, rej = 0;
            for (int k = 0; k < g_B; ++k)
                (rb->results[k].status == PatchStatus::REJECT ? rej : ok)++;
            g_ok += ok; g_reject += rej;
            long r = g_received.fetch_add(1) + 1;
            double lat = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - g_pubTime[s]).count();
            g_permits->release();   // free the slot -> the sender can publish the next batch

            // warmup gate: skip the first g_warmupBatches from the stats
            if (g_warming.load()) {
                if (r >= g_warmupBatches) {
                    g_baseReceived = r;
                    g_baseSent = g_sent.load();
                    g_baseOk = g_ok.load();
                    g_baseReject = g_reject.load();
                    g_baseDrops = g_drops.load();
                    g_latSumMs = 0.0; g_latMinMs = 0.0; g_latMaxMs = 0.0; g_latCount = 0;
                    g_tStart = std::chrono::high_resolution_clock::now();
                    g_warming.store(false);
                    fmt::print("[TEST] Warmup complete ({} batches skipped). Measuring...\n", g_warmupBatches);
                    fflush(stdout);
                }
                continue; // do not accumulate stats during warmup
            }

            // measured window
            if (g_latCount == 0 || lat < g_latMinMs) g_latMinMs = lat;
            if (lat > g_latMaxMs) g_latMaxMs = lat;
            g_latSumMs += lat; ++g_latCount;

            const long mr = r - g_baseReceived; // measured batch index

            if (!g_savedFrame0.exchange(true)) {
                cv::Mat mosaic(g_H * g_B, g_W, CV_8UC3);
                const unsigned char* ovBase = pResImg + IpcResultImageSlotOffset(*cpPtr, (DWORD)s);
                for (int k = 0; k < g_B; ++k) {
                    cv::Mat one(g_H, g_W, CV_8UC3, const_cast<unsigned char*>(ovBase) + (size_t)k * g_imgBytes);
                    one.copyTo(mosaic(cv::Rect(0, k * g_H, g_W, g_H)));
                }
                cv::imwrite("test_overlays_frame0.png", mosaic);
                fmt::print("[TEST] Saved test_overlays_frame0.png\n"); fflush(stdout);
            }

            if (mr % 500 == 0) {
                double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - g_tStart).count();
                const double imgPerS = 1000.0 * mr * g_B / ms;   // frame rate
                const double msPerImg = ms / (mr * g_B);         // throughput time/image
                const double avgLatBatch = g_latSumMs / g_latCount;
                const long mdrops = g_drops.load() - g_baseDrops;
                fmt::print("[TEST] {} batches | {:.1f} img/s | {:.3f} ms/img | {:.2f} ms/batch "
                    "| latency {:.1f} ms/batch ({:.3f} ms/img) | drops {}\n",
                    mr, imgPerS, msPerImg, msPerImg * g_B, avgLatBatch, avgLatBatch / g_B, mdrops);
                fflush(stdout);
            }
        }

        if (g_senderDone.load()) {
            if (!draining) { draining = true; drainDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5); }
            if (g_received.load() >= g_sent.load()) break;
            if (std::chrono::steady_clock::now() > drainDeadline) {
                fmt::print(stderr, "[TEST] drain timeout: {} result(s) never arrived\n",
                    g_sent.load() - g_received.load());
                break;
            }
        }
    }
}

static bool Configure()
{
    if (g_configured.load()) { fmt::print("[TEST] already configured.\n"); return true; }

    WaitForSingleObject(hListMutex, INFINITE);
    pList->numPunti = 1;
    controlPoint& cp = pList->points[0];
    ZeroMemory(&cp, sizeof(cp));
    cp.idPunto = kId;
    cp.sizeX = (DWORD)g_W;
    cp.sizeY = (DWORD)g_H;
    cp.bpp = (DWORD)g_bpp;
	cp.inferenceThreads = (DWORD)g_inferenceThreads;
    cp.batchSize = (DWORD)g_B;
	cp.loopBatch1 = (DWORD)g_loop1;
    cp.ringSlots = (DWORD)g_inFlight;   // the controller imposes the ring depth
    cp.inferenceType = InferenceType::ANOMALY;
    cp.status = PointState::IDLE;
    {
        std::wstring wModel(g_modelPath.begin(), g_modelPath.end());
        wcsncpy_s(cp.pathModello, wModel.c_str(), _TRUNCATE);
        wcsncpy_s(cp.mutexName, kMutexName, _TRUNCATE);
        wcsncpy_s(cp.eventReadyName, kReadyName, _TRUNCATE);
        wcsncpy_s(cp.resultsEventName, kResultsName, _TRUNCATE);
    }
    pList->state = ListState::UPDATE_PENDING;
    ReleaseMutex(hListMutex);
    cpPtr = &pList->points[0];

    fmt::print("[TEST] Triggering configuration ({})...\n", g_modelPath);
    fmt::print("[TEST] Waiting for AI engine (first-time TensorRT build can take minutes). Press 'q' to abort.\n");
    fflush(stdout);
    SetEvent(hTrigger);

    // Unbounded wait for the ack, abortable with 'q'. No fixed timeout: the
    // TensorRT engine build on the first run is legitimately slow.
    bool ack = false, aborted = false;
    while (!ack && !aborted) {
        DWORD w = WaitForSingleObject(hAck, 250);
        if (w == WAIT_OBJECT_0) { ack = true; }
        else if (_kbhit()) {
            int c = _getch();
            if (c == 'q' || c == 'Q') aborted = true;
        }
    }
    if (aborted) { fmt::print("[TEST] Configuration aborted by user.\n"); return false; }

    if (pList->state != ListState::CONFIGURED) {
        fmt::print(stderr, "[TEST] Configuration FAILED (state != CONFIGURED).\n");
        return false;
    }

    // Open the per-point MMFs the worker created during configuration.
    const SIZE_T inBytes = IpcInputMmfBytes(cp);
    const SIZE_T outImgBytes = IpcResultImageMmfBytes(cp);
    hIn = OpenWithRetry([&] { return OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, fmt::format(L"MMF_{}_IMAGE", kId).c_str()); }, "input MMF");
    hResImg = OpenWithRetry([&] { return OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, fmt::format(L"MMF_{}_RESIMAGE", kId).c_str()); }, "result-image MMF");
    hRes = OpenWithRetry([&] { return OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, fmt::format(L"MMF_{}_RESULT", kId).c_str()); }, "result MMF");
    if (!hIn || !hResImg || !hRes) return false;

    pIn = (unsigned char*)MapViewOfFile(hIn, FILE_MAP_WRITE, 0, 0, inBytes);
    pResImg = (unsigned char*)MapViewOfFile(hResImg, FILE_MAP_READ, 0, 0, outImgBytes);
    pRes = (batchResultBlock*)MapViewOfFile(hRes, FILE_MAP_READ, 0, 0, IpcResultBlockMmfBytes(cp));
    if (!pIn || !pResImg || !pRes) { fmt::print(stderr, "[TEST] MapViewOfFile per-point failed\n"); return false; }
    header = reinterpret_cast<batchInputHeader*>(pIn);
    header->publishSeq = 0;

    g_configured.store(true);
    fmt::print("[TEST] CONFIGURED. Press 's' to start inference.\n");
    fflush(stdout);
    return true;
}

static void StartInference()
{
    if (!g_configured.load()) { fmt::print("[TEST] configure first ('c').\n"); return; }
    if (g_infRunning.load()) { fmt::print("[TEST] inference already running.\n"); return; }

    g_stop.store(false); g_senderDone.store(false); g_savedFrame0.store(false);
    g_sent.store(0); g_received.store(0); g_ok.store(0); g_reject.store(0); g_drops.store(0);
    g_imgCursor = 0;
    g_latSumMs = 0.0; g_latMinMs = 0.0; g_latMaxMs = 0.0; g_latCount = 0;
    g_baseReceived = 0; g_baseSent = 0; g_baseOk = 0; g_baseReject = 0; g_baseDrops = 0;
    g_warming.store(g_warmupBatches > 0);

    // Skip any stale RESULT_READY slots left in the ring by a previous run, so
    // they are not re-counted (fresh batches will carry higher seq values).
    for (int s = 0; s < g_inFlight; ++s) {
        auto* rb = reinterpret_cast<const batchResultBlock*>(
            reinterpret_cast<const unsigned char*>(pRes) + (size_t)s * sizeof(batchResultBlock));
        g_lastSeq[s] = (rb->state == InferenceState::RESULT_READY) ? (int)rb->seq : -1;
    }

    g_tStart = std::chrono::high_resolution_clock::now();
    g_receiverThread = std::thread(ReceiverLoop);
    g_senderThread = std::thread(SenderLoop);
    g_infRunning.store(true);
    if (g_frameIntervalMs <= 0)
        fmt::print("[TEST] Inference STARTED | FREE-RUNNING (max throughput, no drops) | "
            "up to {} in flight. Press 'x' to stop.\n", g_inFlight);
    else
        fmt::print("[TEST] Inference STARTED | frame grabber @ {} ms/batch ({:.1f} img/s target) | "
            "up to {} in flight. Press 'x' to stop.\n",
            g_frameIntervalMs, 1000.0 * g_B / g_frameIntervalMs, g_inFlight);
    fflush(stdout);
}

static void StopInference()
{
    if (!g_infRunning.load()) { fmt::print("[TEST] inference not running.\n"); return; }
    g_stop.store(true);
    if (g_senderThread.joinable()) g_senderThread.join();
    if (g_receiverThread.joinable()) g_receiverThread.join();
    g_infRunning.store(false);

    double ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - g_tStart).count();
    // All figures are for the MEASURED window (after warmup): subtract baselines.
    const bool warmupDone = !g_warming.load();
    long total = g_received.load() - g_baseReceived;
    long drops = g_drops.load() - g_baseDrops;
    long msent = g_sent.load() - g_baseSent;
    long okCnt = g_ok.load() - g_baseOk;
    long rejCnt = g_reject.load() - g_baseReject;
    double patches = (double)total * g_B;

    const double batchesPerS = (ms > 0) ? 1000.0 * total / ms : 0.0;
    const double imgPerS = (ms > 0) ? 1000.0 * patches / ms : 0.0;   // frame rate
    const double msPerImg = (patches > 0) ? ms / patches : 0.0;      // throughput time/image
    const double msPerBatch = (total > 0) ? ms / total : 0.0;
    const double avgLatBatch = (g_latCount > 0) ? g_latSumMs / g_latCount : 0.0;

    fmt::print("\n=================================================\n");
    fmt::print("                  RUN SUMMARY\n");
    fmt::print("=================================================\n");
    if (!warmupDone)
        fmt::print("(!) stopped during warmup: figures include warmup batches\n");
    else
        fmt::print("Warmup skipped     : {} batches\n", g_warmupBatches);
    const long ticks = msent + drops;
    if (g_frameIntervalMs <= 0) {
        fmt::print("Mode               : FREE-RUNNING (max throughput)\n");
    }
    else {
        fmt::print("Frame grabber      : {} ms/batch  ({:.1f} img/s requested)\n",
            g_frameIntervalMs, 1000.0 * g_B / g_frameIntervalMs);
        fmt::print("Ticks (ok+drop)    : {}\n", ticks);
        fmt::print("FRAME DROPS        : {}  ({:.2f}% of ticks)\n",
            drops, ticks > 0 ? 100.0 * drops / ticks : 0.0);
    }
    fmt::print("Patches            : {:.0f}  (OK {}, REJECT {})\n", patches, okCnt, rejCnt);
    fmt::print("Run time           : {:.1f} ms\n", ms);
    fmt::print("-------------------------------------------------\n");
    fmt::print("THROUGHPUT (steady state)\n");
    fmt::print("  frame rate       : {:.1f} img/s   ({:.1f} batches/s)\n", imgPerS, batchesPerS);
    fmt::print("  time / image     : {:.3f} ms      (target {:.3f} ms)\n", msPerImg, kTargetMsPerImage);
    fmt::print("  time / batch     : {:.2f} ms       (target {:.2f} ms)\n", msPerBatch, kTargetMsPerImage * g_B);
    if (msPerImg > 0) {
        const double pct = 100.0 * kTargetMsPerImage / msPerImg; // % of the GPU limit reached
        fmt::print("  vs GPU limit     : {:.1f}% of the {:.3f} ms/img ceiling ({})\n",
            pct, kTargetMsPerImage,
            msPerImg <= kTargetMsPerImage * 1.05 ? "AT the limit" : "below the limit");
    }
    fmt::print("-------------------------------------------------\n");
    fmt::print("END-TO-END LATENCY (copy -> result, receiver side)\n");
    fmt::print("  per batch        : avg {:.2f} ms | min {:.2f} | max {:.2f}\n",
        avgLatBatch, g_latMinMs, g_latMaxMs);
    fmt::print("  per image        : avg {:.3f} ms\n", avgLatBatch / g_B);
    fmt::print("=================================================\n");
    fflush(stdout);
}

static void QuitAll()
{
    if (g_infRunning.load()) StopInference();

    fmt::print("[TEST] Sending QUIT...\n"); fflush(stdout);
    WaitForSingleObject(hListMutex, INFINITE);
    pList->state = ListState::QUIT;
    if (g_configured.load()) pList->points[0].status = PointState::QUIT;
    ReleaseMutex(hListMutex);
    if (hCpReady) SetEvent(hCpReady);
    if (hCpResults) SetEvent(hCpResults);
    SetEvent(hTrigger);
    WaitForSingleObject(hAck, 30000);
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        fmt::print(stderr, "Usage: {} <onnx_model_path> <images_dir> [W=256] [H=256] [frameIntervalMs=40] [warmupBatches=100] [inFlight=5, max {}] [inferenceThreads=1] [batchSize=17] [loop1=0]\n", argv[0], (int)kMaxRingSlots);
        return -1;
    }
    g_modelPath = argv[1];
    g_imagesDir = argv[2];
    g_W = (argc > 3) ? std::atoi(argv[3]) : 256;
    g_H = (argc > 4) ? std::atoi(argv[4]) : 256;
    if (argc > 5) g_frameIntervalMs = std::atoi(argv[5]);           // 0 = free-running
    if (argc > 6) { int v = std::atoi(argv[6]); if (v >= 0) g_warmupBatches = v; }
    if (argc > 7) {
        int v = std::atoi(argv[7]);
        if (v < 1) v = 1;
        if (v > (int)kMaxRingSlots) v = (int)kMaxRingSlots; // clamp to the ABI ceiling
        g_inFlight = v;
    }
    if (argc > 8) { int v = std::atoi(argv[8]); g_inferenceThreads = (v >= 1) ? v : 1; } 
    if (argc > 9) { int v = std::atoi(argv[9]); g_B = (v >= 1) ? v : 1; }
    if (argc > 10) g_loop1 = (std::atoi(argv[10]) != 0) ? 1 : 0;
    g_permits = std::make_unique<std::counting_semaphore<kMaxRingSlots>>(g_inFlight);
    g_channels = g_bpp / 8;
    g_imgBytes = (size_t)g_W * g_H * g_channels;

    // ---- load every image in the folder, resized to WxH, BGR 8UC3 ----
    if (!fs::exists(g_imagesDir) || !fs::is_directory(g_imagesDir)) {
        fmt::print(stderr, "[TEST] '{}' is not a directory\n", g_imagesDir); return -1;
    }
    for (const auto& e : fs::directory_iterator(g_imagesDir)) {
        if (!e.is_regular_file() || !IsImageFile(e.path())) continue;
        cv::Mat m = cv::imread(e.path().string(), cv::IMREAD_COLOR);
        if (m.empty()) continue;
        cv::Mat r;
        cv::resize(m, r, cv::Size(g_W, g_H));
        if (r.type() != CV_8UC3) r.convertTo(r, CV_8UC3);
        if (!r.isContinuous()) r = r.clone();
        g_images.push_back(r);
    }
    if (g_images.empty()) { fmt::print(stderr, "[TEST] no images found in {}\n", g_imagesDir); return -1; }
    fmt::print("[TEST] Loaded {} images from {} ({}x{}).\n", g_images.size(), g_imagesDir, g_W, g_H);

    // global IPC objects (create-or-open, so start order is free)
    hList = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(controlPointsList), L"CONTROLPOINTLIST");
    if (!hList) { fmt::print(stderr, "CreateFileMapping list failed ({})\n", GetLastError()); return -1; }
    pList = (PTcontrolPointsList)MapViewOfFile(hList, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(controlPointsList));
    if (!pList) { fmt::print(stderr, "MapViewOfFile list failed\n"); return -1; }

    hListMutex = CreateMutex(NULL, FALSE, L"LISTMUTEX");
    hTrigger   = CreateEvent(NULL, FALSE, FALSE, L"LISTEVENTTRIGGER"); // auto-reset
    hAck       = CreateEvent(NULL, FALSE, FALSE, L"LISTEVENTACK");     // auto-reset

    // per-point sync objects (the worker opens these by name during config)
    hCpMutex   = CreateMutex(NULL, FALSE, kMutexName);
    hCpReady   = CreateEvent(NULL, FALSE, FALSE, kReadyName);
    hCpResults = CreateEvent(NULL, FALSE, FALSE, kResultsName);
    if (!hListMutex || !hTrigger || !hAck || !hCpMutex || !hCpReady || !hCpResults) {
        fmt::print(stderr, "[TEST] IPC object creation failed ({})\n", GetLastError()); return -1;
    }

    SetConsoleCtrlHandler(CtrlHandler, TRUE);

    fmt::print("\n=== inference-gpu-test ===\n");
    fmt::print("Commands:  c = configure   s = start inference   x = stop   q = quit\n");
    fmt::print("(start inference-gpu.exe as administrator; any launch order is fine)\n\n");
    fflush(stdout);

    bool running = true;
    while (running) {
        if (g_quitRequested.load()) { QuitAll(); break; }
        if (_kbhit()) {
            int c = _getch();
            switch (tolower(c)) {
            case 'c': Configure(); break;
            case 's': StartInference(); break;
            case 'x': StopInference(); break;
            case 'q': QuitAll(); running = false; break;
            default: break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    // Cleanup
    if (pIn) UnmapViewOfFile(pIn);
    if (pResImg) UnmapViewOfFile(pResImg);
    if (pRes) UnmapViewOfFile(pRes);
    if (pList) UnmapViewOfFile(pList);
    for (HANDLE h : { hIn, hResImg, hRes, hList, hCpMutex, hCpReady, hCpResults, hListMutex, hTrigger, hAck })
        if (h) CloseHandle(h);
    fmt::print("[TEST] Done.\n");
    return 0;
}
