import 'dart:async';
import 'dart:convert';

import 'package:excel/excel.dart' hide Border;
import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../api/esp_api.dart';
import '../ui/ui_card.dart';
import '../utils/save_bytes.dart';

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

  final Excel excel = Excel.createExcel();
  late final Sheet sheet;
  final List<List<String>> csvRows = [
    ["seq", "time", "channel", "target_deg", "actual_deg", "error_deg"],
  ];

  late final ScrollController _verticalController;
  late final ScrollController _horizontalController;

  @override
  void initState() {
    super.initState();

    _verticalController = ScrollController();
    _horizontalController = ScrollController();

    sheet = excel['ServoLog'];
    sheet.appendRow([
      TextCellValue("Seq"),
      TextCellValue("Time"),
      TextCellValue("Channel"),
      TextCellValue("Target (deg)"),
      TextCellValue("Actual (deg)"),
      TextCellValue("Error (deg)"),
    ]);

    sub = WsControlApi.stream().listen((msg) {
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

      // 只有 seq 遞增才記錄
      if (lastSeq != null && seq <= lastSeq!) return;
      lastSeq = seq;

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

      final now = DateTime.now().toIso8601String();

      for (int i = 0; i < len; i++) {
        sheet.appendRow([
          IntCellValue(seq),
          TextCellValue(now),
          TextCellValue("CH${i + 1}"),
          DoubleCellValue(t[i]),
          DoubleCellValue(a[i]),
          DoubleCellValue(e[i]),
        ]);
        csvRows.add([
          seq.toString(),
          now,
          "CH${i + 1}",
          t[i].toStringAsFixed(3),
          a[i].toStringAsFixed(3),
          e[i].toStringAsFixed(3),
        ]);
      }

      if (!mounted) return;

      setState(() {
        target = t.take(len).toList();
        actual = a.take(len).toList();
        error = e.take(len).toList();
        logCount++;
      });
    });
  }

  @override
  void dispose() {
    sub?.cancel();
    _verticalController.dispose();
    _horizontalController.dispose();
    super.dispose();
  }

  Future<void> exportExcel() async {
    if (kIsWeb) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Web 暫不支援匯出")),
      );
      return;
    }

    if (logCount <= 0) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("目前沒有可匯出的資料")),
      );
      return;
    }

    final encoded = excel.encode();
    if (encoded == null) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Excel 產生失敗")),
      );
      return;
    }

    final filename = "servo_log_${DateTime.now().millisecondsSinceEpoch}.xlsx";

    try {
      final path = await FilePicker.platform.saveFile(
        dialogTitle: "儲存 Servo Log",
        fileName: filename,
        type: FileType.custom,
        allowedExtensions: ['xlsx'],
        bytes: kIsWeb ? Uint8List.fromList(encoded) : null,
      );

      if (!mounted) return;

      if (path == null) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("已取消儲存")),
        );
        return;
      }

      if (!kIsWeb) {
        await writeBytesToPath(path, Uint8List.fromList(encoded));
      }

      if (!mounted) return;

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("已匯出：$path")),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("匯出失敗：$e")),
      );
    }
  }

  String _csvEscape(String value) {
    if (!value.contains(',') && !value.contains('"') && !value.contains('\n')) {
      return value;
    }
    return '"${value.replaceAll('"', '""')}"';
  }

  Future<void> exportCsv() async {
    if (logCount <= 0) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("No servo data to export")),
      );
      return;
    }

    final text = csvRows.map((row) => row.map(_csvEscape).join(',')).join('\n');
    final bytes = Uint8List.fromList(utf8.encode(text));
    final filename = "servo_log_${DateTime.now().millisecondsSinceEpoch}.csv";

    try {
      final path = await FilePicker.platform.saveFile(
        dialogTitle: "Save Servo CSV",
        fileName: filename,
        type: FileType.custom,
        allowedExtensions: ['csv'],
        bytes: kIsWeb ? bytes : null,
      );

      if (!mounted) return;

      if (path == null) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("CSV export canceled")),
        );
        return;
      }

      if (!kIsWeb) {
        await writeBytesToPath(path, bytes);
      }

      if (!mounted) return;

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Saved CSV: $path")),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("CSV export failed: $e")),
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
                                        Text(target[i].toStringAsFixed(2)),
                                      ),
                                      DataCell(
                                        Text(actual[i].toStringAsFixed(2)),
                                      ),
                                      DataCell(
                                        Text(error[i].toStringAsFixed(2)),
                                      ),
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
              onPressed: exportExcel,
              child: const Text("匯出 Excel"),
            ),
            ElevatedButton(
              onPressed: exportCsv,
              child: const Text("匯出 CSV"),
            ),
            Text("已記錄 $logCount 筆"),
          ],
        ),
      ],
    );

    if (widget.compact) {
      return content;
    }

    return UiCard(
      title: "Servo 狀態",
      minHeight: 340,
      child: content,
    );
  }
}
