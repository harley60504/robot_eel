import 'dart:convert';
import 'package:http/http.dart' as http;
import '../config.dart';

class GaitState {
  final String current;
  final List<Map<String, dynamic>> gaits;

  const GaitState({
    required this.current,
    required this.gaits,
  });
}

class PythonApi {
  static int get port => ApiConfig.pythonPort;

  static Uri _u(String host, String path) =>
      Uri.parse("http://$host:$port$path");

  static Future<bool> ping({required String pcHost}) async {
    try {
      final res = await http
          .get(_u(pcHost, "/"))
          .timeout(const Duration(milliseconds: 700));
      if (res.statusCode != 200) return false;

      final data = jsonDecode(res.body);
      return data is Map && data.containsKey("preview_running");
    } catch (_) {
      return false;
    }
  }

  static Future<Map<String, dynamic>?> status({required String pcHost}) async {
    try {
      final res = await http
          .get(_u(pcHost, "/"))
          .timeout(const Duration(milliseconds: 700));
      if (res.statusCode != 200) return null;

      final data = jsonDecode(res.body);
      return data is Map<String, dynamic> ? data : null;
    } catch (_) {
      return null;
    }
  }

  static Future<bool> waitUntilReady({
    required String pcHost,
    Duration timeout = const Duration(seconds: 8),
  }) async {
    final deadline = DateTime.now().add(timeout);
    while (DateTime.now().isBefore(deadline)) {
      if (await ping(pcHost: pcHost)) return true;
      await Future.delayed(const Duration(milliseconds: 350));
    }
    return false;
  }

  // ===============================
  // ESP Host
  // ===============================
  static Future<bool> setEspHost({
    required String pcHost,
    required String espHost,
  }) async {
    try {
      final res = await http
          .post(
            _u(pcHost, "/set_esp_host"),
            headers: {"Content-Type": "application/json"},
            body: jsonEncode({
              "esp_host": espHost,
            }),
          )
          .timeout(const Duration(milliseconds: 900));
      return res.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  // ===============================
  // Start / Stop
  // ===============================
  static Future<bool> start({required String pcHost}) async {
    try {
      final res = await http
          .post(_u(pcHost, "/start"))
          .timeout(const Duration(milliseconds: 900));
      return res.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  static Future<bool> stop({required String pcHost}) async {
    try {
      final res = await http
          .post(_u(pcHost, "/stop"))
          .timeout(const Duration(milliseconds: 900));
      return res.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  static Future<List<Map<String, dynamic>>> gaits({
    required String pcHost,
  }) async {
    final state = await gaitState(pcHost: pcHost);
    return state?.gaits ?? const [];
  }

  static Future<GaitState?> gaitState({
    required String pcHost,
  }) async {
    try {
      final res = await http
          .get(_u(pcHost, "/gaits"))
          .timeout(const Duration(milliseconds: 900));
      if (res.statusCode != 200) return null;
      final data = jsonDecode(res.body);
      if (data is! Map || data["gaits"] is! List) return null;
      return GaitState(
        current: data["current"]?.toString() ?? "",
        gaits: List<Map<String, dynamic>>.from(data["gaits"]),
      );
    } catch (_) {
      return null;
    }
  }

  static Future<bool> setGait({
    required String pcHost,
    required String gait,
  }) async {
    try {
      final res = await http
          .post(
            _u(pcHost, "/set_gait"),
            headers: {"Content-Type": "application/json"},
            body: jsonEncode({"gait": gait}),
          )
          .timeout(const Duration(milliseconds: 900));
      if (res.statusCode != 200) return false;
      final data = jsonDecode(res.body);
      return data is Map ? data["ok"] == true : true;
    } catch (_) {
      return false;
    }
  }

  static Future<bool> setOutputMode({
    required String pcHost,
    required String outputMode,
  }) async {
    try {
      final res = await http
          .post(
            _u(pcHost, "/set_output_mode"),
            headers: {"Content-Type": "application/json"},
            body: jsonEncode({"output_mode": outputMode}),
          )
          .timeout(const Duration(milliseconds: 900));
      return res.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  // ===============================
  // RTT Measure
  // ===============================
  static Future<bool> measureOn({required String pcHost}) async {
    try {
      final res = await http.post(_u(pcHost, "/measure_on"));
      return res.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  static Future<bool> measureOff({required String pcHost}) async {
    try {
      final res = await http.post(_u(pcHost, "/measure_off"));
      return res.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  static Future<bool> setRecorderUrl({
    required String pcHost,
    required String recorderUrl,
  }) async {
    try {
      final res = await http.post(
        _u(pcHost, "/settings/recorder_url"),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({"recorder_url": recorderUrl}),
      );
      return res.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  static Future<bool> recordingStart({required String pcHost}) async {
    try {
      final res = await http.post(_u(pcHost, "/recording/start"));
      return res.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  static Future<bool> recordingStop({required String pcHost}) async {
    try {
      final res = await http.post(_u(pcHost, "/recording/stop"));
      return res.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  static Future<bool> setRecordingTelemetry({
    required String pcHost,
    required bool servo,
    required bool imu,
  }) async {
    try {
      final res = await http.post(
        _u(pcHost, "/settings/recording_telemetry"),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({"servo": servo, "imu": imu}),
      );
      return res.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  static Future<String?> downloadTelemetryCsv({
    required String pcHost,
    required String kind,
  }) async {
    try {
      final res = await http
          .post(_u(pcHost, "/telemetry/download_csv/$kind"))
          .timeout(const Duration(seconds: 45));
      if (res.statusCode != 200) return null;
      final data = jsonDecode(res.body);
      if (data is! Map || data["ok"] != true) return null;
      return data["csv_path"]?.toString();
    } catch (_) {
      return null;
    }
  }

  static Future<bool> clearTelemetryLogs({required String pcHost}) async {
    try {
      final res = await http
          .post(_u(pcHost, "/telemetry/clear_all"))
          .timeout(const Duration(seconds: 5));
      if (res.statusCode != 200) return false;
      final data = jsonDecode(res.body);
      return data is Map ? data["ok"] == true : true;
    } catch (_) {
      return false;
    }
  }

  static Uri previewFrameUri({required String pcHost}) =>
      _u(pcHost, "/preview.jpg");

  static Future<bool> previewStart({required String pcHost}) async {
    try {
      final res = await http.post(_u(pcHost, "/preview/start"));
      return res.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  static Future<bool> previewStop({required String pcHost}) async {
    try {
      final res = await http.post(_u(pcHost, "/preview/stop"));
      return res.statusCode == 200;
    } catch (_) {
      return false;
    }
  }
}
