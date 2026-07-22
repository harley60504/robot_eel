import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

import '../api/python_api.dart';
import '../bridge/python_bridge.dart';
import '../bridge/python_process_launcher.dart';
import '../config.dart';

class PythonRtspPreview extends StatefulWidget {
  const PythonRtspPreview({super.key});

  @override
  State<PythonRtspPreview> createState() => _PythonRtspPreviewState();
}

class _PythonRtspPreviewState extends State<PythonRtspPreview> {
  Timer? frameTimer;
  Timer? statusTimer;
  Uint8List? frame;
  String status = "RTSP preview stopped";
  bool running = false;
  bool busy = false;
  bool recording = false;
  bool recordingBusy = false;
  bool exportServoCsv = true;
  bool exportImuCsv = true;
  bool telemetryBusy = false;
  bool fetchingFrame = false;
  int frames = 0;
  double localFps = 0;
  double streamFps = 0;
  DateTime lastFps = DateTime.now();

  @override
  void initState() {
    super.initState();
    ensurePythonReady().then((ready) {
      if (!mounted || !ready) return;
      refreshPreviewStatus();
      statusTimer = Timer.periodic(const Duration(seconds: 1), (_) {
        refreshPreviewStatus();
      });
      setState(() => status = "RTSP preview stopped");
    });
  }

  @override
  void dispose() {
    frameTimer?.cancel();
    statusTimer?.cancel();
    PythonApi.previewStop(pcHost: ApiConfig.pythonHost);
    super.dispose();
  }

  Future<bool> ensurePythonReady() async {
    final pcHost = ApiConfig.pythonHost;
    var ready = await PythonApi.ping(pcHost: pcHost);
    if (ready) return true;

    setState(() => status = "Starting Python backend...");
    final launch = await PythonProcessLauncher.launch();
    if (!launch.ok) {
      setState(() => status = launch.message);
      return false;
    }

    ready = await PythonApi.waitUntilReady(pcHost: pcHost);
    if (!ready) setState(() => status = "Python backend offline");
    return ready;
  }

  Future<void> startPreview() async {
    if (busy) return;
    setState(() {
      busy = true;
      status = "Starting RTSP preview...";
    });

    final ready = await ensurePythonReady();
    if (!ready) {
      if (mounted) setState(() => busy = false);
      return;
    }

    final pcHost = ApiConfig.pythonHost;
    await PythonBridge.syncEsp32HostToPython(pcHost: pcHost);
    await PythonApi.setRecorderUrl(
      pcHost: pcHost,
      recorderUrl: ApiConfig.recorderUrl,
    );

    final ok = await PythonApi.previewStart(pcHost: pcHost);
    if (!mounted) return;
    setState(() {
      running = ok;
      busy = false;
      status = ok ? "Waiting for RTSP frame..." : "RTSP preview failed";
    });

    if (ok) {
      frameTimer?.cancel();
      statusTimer?.cancel();
      scheduleNextFrame();
      statusTimer = Timer.periodic(const Duration(seconds: 1), (_) {
        refreshPreviewStatus();
      });
    }
  }

  Future<void> stopPreview() async {
    frameTimer?.cancel();
    statusTimer?.cancel();
    await PythonApi.previewStop(pcHost: ApiConfig.pythonHost);
    if (!mounted) return;
    setState(() {
      running = false;
      localFps = 0;
      streamFps = 0;
      status = "RTSP preview stopped";
    });
  }

  Future<void> togglePreview() async {
    if (running) {
      await stopPreview();
    } else {
      await startPreview();
    }
  }

  Future<void> toggleRecording() async {
    if (recordingBusy) return;
    setState(() => recordingBusy = true);

    final ready = await ensurePythonReady();
    if (!ready) {
      if (mounted) setState(() => recordingBusy = false);
      return;
    }

    final pcHost = ApiConfig.pythonHost;
    await PythonBridge.syncEsp32HostToPython(pcHost: pcHost);
    await PythonApi.setRecorderUrl(
      pcHost: pcHost,
      recorderUrl: ApiConfig.recorderUrl,
    );
    await PythonApi.setRecordingTelemetry(
      pcHost: pcHost,
      servo: exportServoCsv,
      imu: exportImuCsv,
    );

    final ok = recording
        ? await PythonApi.recordingStop(pcHost: pcHost)
        : await PythonApi.recordingStart(pcHost: pcHost);

    if (!mounted) return;
    setState(() {
      if (ok) recording = !recording;
      recordingBusy = false;
      status = ok
          ? (recording ? "RTSP preview recording" : "RTSP preview")
          : (recording ? "Stop recording failed" : "Start recording failed");
    });
  }

  Future<void> fetchFrame() async {
    if (fetchingFrame) return;
    fetchingFrame = true;
    try {
      final res = await http
          .get(PythonApi.previewFrameUri(pcHost: ApiConfig.pythonHost))
          .timeout(const Duration(milliseconds: 650));
      if (!mounted || res.statusCode != 200 || res.bodyBytes.isEmpty) return;

      setState(() {
        frame = res.bodyBytes;
        status = "RTSP preview";
      });
      calcFps();
    } catch (_) {
    } finally {
      fetchingFrame = false;
      if (mounted && running) scheduleNextFrame();
    }
  }

  void scheduleNextFrame() {
    frameTimer?.cancel();
    final targetFps = streamFps > 1 ? streamFps : 30.0;
    final intervalMs = (1000 / targetFps).round().clamp(16, 200);
    frameTimer = Timer(Duration(milliseconds: intervalMs), fetchFrame);
  }

  Future<void> refreshPreviewStatus() async {
    final data = await PythonApi.status(pcHost: ApiConfig.pythonHost);
    if (!mounted || data == null) return;

    final backendFps = (data["preview_fps"] as num?)?.toDouble() ?? 0.0;
    final isRunning = data["preview_running"] == true;
    setState(() {
      streamFps = backendFps;
      running = isRunning;
      exportServoCsv = data["record_export_servo_csv"] != false;
      exportImuCsv = data["record_export_imu_csv"] != false;
    });
  }

  Future<void> setTelemetryExport({
    required bool servo,
    required bool imu,
  }) async {
    setState(() {
      exportServoCsv = servo;
      exportImuCsv = imu;
    });
    await PythonApi.setRecordingTelemetry(
      pcHost: ApiConfig.pythonHost,
      servo: exportServoCsv,
      imu: exportImuCsv,
    );
  }

  Future<void> downloadTelemetryCsv(String kind) async {
    if (telemetryBusy) return;
    setState(() {
      telemetryBusy = true;
      status = "Downloading $kind CSV...";
    });

    final ready = await ensurePythonReady();
    if (!ready) {
      if (mounted) setState(() => telemetryBusy = false);
      return;
    }

    final pcHost = ApiConfig.pythonHost;
    await PythonBridge.syncEsp32HostToPython(pcHost: pcHost);

    final path = await PythonApi.downloadTelemetryCsv(
      pcHost: pcHost,
      kind: kind,
    );

    if (!mounted) return;
    setState(() {
      telemetryBusy = false;
      status = path == null ? "Download $kind CSV failed" : "Saved $kind CSV";
    });

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(path ?? "Download $kind CSV failed"),
      ),
    );
  }

  Future<void> clearTelemetryLogs() async {
    if (telemetryBusy) return;
    setState(() {
      telemetryBusy = true;
      status = "Clearing telemetry logs...";
    });

    final ready = await ensurePythonReady();
    if (!ready) {
      if (mounted) setState(() => telemetryBusy = false);
      return;
    }

    final pcHost = ApiConfig.pythonHost;
    await PythonBridge.syncEsp32HostToPython(pcHost: pcHost);

    final ok = await PythonApi.clearTelemetryLogs(
      pcHost: pcHost,
    );

    if (!mounted) return;
    setState(() {
      telemetryBusy = false;
      status = ok ? "Telemetry logs cleared" : "Clear telemetry failed";
    });

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(ok ? "Telemetry logs cleared" : "Clear telemetry failed"),
      ),
    );
  }

  void calcFps() {
    frames++;
    final now = DateTime.now();
    final diff = now.difference(lastFps).inMilliseconds;
    if (diff >= 1000) {
      setState(() {
        localFps = frames * 1000 / diff;
        frames = 0;
        lastFps = now;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(12),
      child: Stack(
        fit: StackFit.expand,
        children: [
          Container(
            color: const Color(0xFF151A20),
            child: Center(
              child: frame == null
                  ? Text(status, style: const TextStyle(color: Colors.white70))
                  : FittedBox(
                      fit: BoxFit.contain,
                      child: Image.memory(frame!, gaplessPlayback: true),
                    ),
            ),
          ),
          Positioned(
            top: 8,
            left: 8,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                "RTSP FPS: ${(streamFps > 0 ? streamFps : localFps).toStringAsFixed(1)}",
                style: const TextStyle(color: Colors.white),
              ),
            ),
          ),
          Positioned(
            top: 8,
            right: 8,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Wrap(
                spacing: 6,
                runSpacing: 6,
                crossAxisAlignment: WrapCrossAlignment.center,
                alignment: WrapAlignment.end,
                children: [
                  IconButton.filledTonal(
                    tooltip: running ? "Stop preview" : "Start preview",
                    onPressed: busy ? null : togglePreview,
                    icon: Icon(running ? Icons.videocam_off : Icons.videocam),
                  ),
                  IconButton.filled(
                    tooltip: recording ? "Stop recording" : "Start recording",
                    onPressed: recordingBusy ? null : toggleRecording,
                    style: IconButton.styleFrom(
                      backgroundColor: recording ? Colors.redAccent : null,
                      foregroundColor: Colors.white,
                    ),
                    icon: Icon(
                      recording ? Icons.stop : Icons.fiber_manual_record,
                    ),
                  ),
                  _TelemetryPill(
                    label: "Servo",
                    selected: exportServoCsv,
                    icon: Icons.table_chart,
                    onToggle: (value) => setTelemetryExport(
                      servo: value,
                      imu: exportImuCsv,
                    ),
                    onDownload: telemetryBusy
                        ? null
                        : () => downloadTelemetryCsv("servo"),
                  ),
                  _TelemetryPill(
                    label: "IMU",
                    selected: exportImuCsv,
                    icon: Icons.sensors,
                    onToggle: (value) => setTelemetryExport(
                      servo: exportServoCsv,
                      imu: value,
                    ),
                    onDownload: telemetryBusy
                        ? null
                        : () => downloadTelemetryCsv("imu"),
                  ),
                  IconButton.filledTonal(
                    tooltip: "Clear telemetry logs",
                    onPressed: telemetryBusy ? null : clearTelemetryLogs,
                    icon: const Icon(Icons.cleaning_services),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _TelemetryPill extends StatelessWidget {
  final String label;
  final bool selected;
  final IconData icon;
  final ValueChanged<bool> onToggle;
  final VoidCallback? onDownload;

  const _TelemetryPill({
    required this.label,
    required this.selected,
    required this.icon,
    required this.onToggle,
    required this.onDownload,
  });

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: selected
            ? Theme.of(context)
                .colorScheme
                .primaryContainer
                .withValues(alpha: 0.72)
            : const Color(0xFF1A2027),
        borderRadius: BorderRadius.circular(28),
        border: Border.all(color: Colors.white10),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Tooltip(
            message: "Record $label CSV",
            child: IconButton(
              visualDensity: VisualDensity.compact,
              onPressed: () => onToggle(!selected),
              icon: Icon(
                selected ? Icons.check : Icons.close,
                size: 18,
              ),
            ),
          ),
          Tooltip(
            message: "Download $label CSV",
            child: TextButton.icon(
              onPressed: onDownload,
              icon: Icon(icon, size: 18),
              label: Text(label),
              style: TextButton.styleFrom(
                foregroundColor: Colors.white,
                padding: const EdgeInsets.only(left: 0, right: 10),
                minimumSize: const Size(0, 40),
                tapTargetSize: MaterialTapTargetSize.shrinkWrap,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
