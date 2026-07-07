import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';

import '../api/esp_api.dart';
import '../api/esp_http_api.dart';
import '../ui/ui_card.dart';
import '../utils/save_csv_file.dart';

class ServoTable extends StatefulWidget {
  final bool compact;

  const ServoTable({super.key, this.compact = false});

  @override
  State<ServoTable> createState() => _ServoTableState();
}

class _ServoTableState extends State<ServoTable> {
  List<double> target = [];
  List<double> actual = [];
  List<double> error = [];

  int? lastSeq;
  int logCount = 0;

  StreamSubscription? sub;

  late final ScrollController _verticalController;
  late final ScrollController _horizontalController;

  @override
  void initState() {
    super.initState();

    _verticalController = ScrollController();
    _horizontalController = ScrollController();

    sub = WsControlApi.stream().listen(_handleWsMessage);
  }

  void _handleWsMessage(dynamic msg) {
    if (!mounted) return;

    dynamic data = msg;

    if (msg is String) {
      try {
        data = jsonDecode(msg);
      } catch (_) {
        return;
      }
    }

    if (data is! Map) return;
    if (data["type"] != "servo_status") return;

    final int seq = (data["seq"] is num) ? (data["seq"] as num).toInt() : -1;
    if (lastSeq != null && seq >= 0 && seq <= lastSeq!) return;
    if (seq >= 0) lastSeq = seq;

    final rawTarget = data["target"];
    final rawActual = data["actual"];
    final rawError = data["error"];

    if (rawTarget is! List || rawActual is! List || rawError is! List) {
      return;
    }

    late final List<double> t;
    late final List<double> a;
    late final List<double> e;

    try {
      t = rawTarget.map((v) => (v as num).toDouble()).toList();
      a = rawActual.map((v) => (v as num).toDouble()).toList();
      e = rawError.map((v) => (v as num).toDouble()).toList();
    } catch (_) {
      return;
    }

    final int len = [t.length, a.length, e.length, 6].reduce(
      (x, y) => x < y ? x : y,
    );

    if (len <= 0) return;

    if (!mounted) return;

    setState(() {
      target = t.take(len).toList();
      actual = a.take(len).toList();
      error = e.take(len).toList();
      logCount++;
    });
  }

  @override
  void dispose() {
    sub?.cancel();
    _verticalController.dispose();
    _horizontalController.dispose();
    super.dispose();
  }

  Future<void> exportCsv() async {
    final filename = "servo_log_${DateTime.now().millisecondsSinceEpoch}.csv";

    try {
      final bytes = await EspHttpApi.servoLogCsv();
      int? samples;
      try {
        samples = await EspHttpApi.servoLogStatus();
      } catch (_) {
        samples = null;
      }
      final path = await saveCsvFile(filename, bytes);

      if (!mounted) return;

      if (path == null) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("CSV export canceled")),
        );
        return;
      }

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            samples == null
                ? "Saved CSV: $path"
                : "Saved CSV: $path ($samples samples)",
          ),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("ESP32 CSV download failed: $e")),
      );
    }
  }

  Future<void> clearServoCache() async {
    try {
      final ok = await EspHttpApi.clearServoLog();
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
            content: Text(ok ? "ESP32 servo cache cleared" : "Clear failed")),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Clear failed: $e")),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final int n = [target.length, actual.length, error.length].reduce(
      (a, b) => a < b ? a : b,
    );
    final int visibleRows = n > 6 ? 6 : n;

    final content = Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        SizedBox(
          height: widget.compact ? 196 : 220,
          width: double.infinity,
          child: Container(
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(8),
              border: Border.all(
                color: Theme.of(context).dividerColor,
                width: 1,
              ),
            ),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: Scrollbar(
                controller: _verticalController,
                thumbVisibility: true,
                child: SingleChildScrollView(
                  controller: _verticalController,
                  scrollDirection: Axis.vertical,
                  child: Scrollbar(
                    controller: _horizontalController,
                    thumbVisibility: true,
                    notificationPredicate: (_) => false,
                    child: SingleChildScrollView(
                      controller: _horizontalController,
                      scrollDirection: Axis.horizontal,
                      child: ConstrainedBox(
                        constraints: BoxConstraints(
                          minWidth: widget.compact ? 320 : 480,
                        ),
                        child: DataTable(
                          columnSpacing: widget.compact ? 8 : 16,
                          horizontalMargin: widget.compact ? 6 : 12,
                          headingRowHeight: widget.compact ? 34 : 44,
                          dataRowMinHeight: widget.compact ? 32 : 40,
                          dataRowMaxHeight: widget.compact ? 32 : 40,
                          columns: const [
                            DataColumn(label: Text("CH")),
                            DataColumn(label: Text("Target")),
                            DataColumn(label: Text("Actual")),
                            DataColumn(label: Text("Error")),
                          ],
                          rows: visibleRows == 0
                              ? const [
                                  DataRow(
                                    cells: [
                                      DataCell(Text("-")),
                                      DataCell(Text("-")),
                                      DataCell(Text("-")),
                                      DataCell(Text("-")),
                                    ],
                                  ),
                                ]
                              : List.generate(visibleRows, (i) {
                                  return DataRow(
                                    cells: [
                                      DataCell(Text("${i + 1}")),
                                      DataCell(
                                          Text(target[i].toStringAsFixed(2))),
                                      DataCell(
                                          Text(actual[i].toStringAsFixed(2))),
                                      DataCell(
                                          Text(error[i].toStringAsFixed(2))),
                                    ],
                                  );
                                }),
                        ),
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ),
        ),
        const SizedBox(height: 12),
        Wrap(
          spacing: 12,
          runSpacing: 8,
          crossAxisAlignment: WrapCrossAlignment.center,
          children: [
            ElevatedButton(
              onPressed: exportCsv,
              child: const Text("Download CSV"),
            ),
            OutlinedButton(
              onPressed: clearServoCache,
              child: const Text("Clear ESP cache"),
            ),
            Text("Log frames: $logCount"),
          ],
        ),
      ],
    );

    if (widget.compact) {
      return content;
    }

    return UiCard(
      title: "Servo Status",
      minHeight: 340,
      child: content,
    );
  }
}
