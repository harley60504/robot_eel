import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';

import '../api/esp_api.dart';
import '../ui/ui_card.dart';

class ImuStatusPanel extends StatefulWidget {
  final bool compact;

  const ImuStatusPanel({super.key, this.compact = false});

  @override
  State<ImuStatusPanel> createState() => _ImuStatusPanelState();
}

class _ImuStatusPanelState extends State<ImuStatusPanel> {
  StreamSubscription? sub;

  int? seq;
  int? tMs;
  double? tempC;
  List<double> accel = const [];
  List<double> gyro = const [];
  int frames = 0;

  @override
  void initState() {
    super.initState();
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
    if (data["type"] != "imu_status") return;

    final rawAccel = data["accel"];
    final rawGyro = data["gyro"];
    if (rawAccel is! List || rawGyro is! List) return;

    late final List<double> parsedAccel;
    late final List<double> parsedGyro;
    try {
      parsedAccel = rawAccel.map((v) => (v as num).toDouble()).take(3).toList();
      parsedGyro = rawGyro.map((v) => (v as num).toDouble()).take(3).toList();
    } catch (_) {
      return;
    }

    if (parsedAccel.length < 3 || parsedGyro.length < 3) return;

    setState(() {
      seq = (data["seq"] is num) ? (data["seq"] as num).toInt() : seq;
      tMs = (data["t_ms"] is num) ? (data["t_ms"] as num).toInt() : tMs;
      tempC =
          (data["tempC"] is num) ? (data["tempC"] as num).toDouble() : tempC;
      accel = parsedAccel;
      gyro = parsedGyro;
      frames++;
    });
  }

  @override
  void dispose() {
    sub?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final content = Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        _MetricGrid(
          accel: accel,
          gyro: gyro,
          tempC: tempC,
          compact: widget.compact,
        ),
        const SizedBox(height: 10),
        Wrap(
          spacing: 12,
          runSpacing: 6,
          children: [
            Text("Frames: $frames"),
            Text("Seq: ${seq ?? "-"}"),
            Text("t_ms: ${tMs ?? "-"}"),
          ],
        ),
      ],
    );

    if (widget.compact) {
      return UiCard(
        title: "IMU Status",
        minHeight: 170,
        child: content,
      );
    }

    return UiCard(
      title: "IMU Status",
      minHeight: 220,
      child: content,
    );
  }
}

class _MetricGrid extends StatelessWidget {
  final List<double> accel;
  final List<double> gyro;
  final double? tempC;
  final bool compact;

  const _MetricGrid({
    required this.accel,
    required this.gyro,
    required this.tempC,
    required this.compact,
  });

  @override
  Widget build(BuildContext context) {
    final rows = [
      ("Accel X", _value(accel, 0, 3)),
      ("Accel Y", _value(accel, 1, 3)),
      ("Accel Z", _value(accel, 2, 3)),
      ("Gyro X", _value(gyro, 0, 2)),
      ("Gyro Y", _value(gyro, 1, 2)),
      ("Gyro Z", _value(gyro, 2, 2)),
      ("Temp", tempC == null ? "-" : "${tempC!.toStringAsFixed(1)} C"),
    ];

    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: rows.map((row) {
        return SizedBox(
          width: compact ? 104 : 120,
          child: DecoratedBox(
            decoration: BoxDecoration(
              border: Border.all(color: Theme.of(context).dividerColor),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    row.$1,
                    style: Theme.of(context).textTheme.labelSmall,
                  ),
                  const SizedBox(height: 4),
                  Text(
                    row.$2,
                    style: const TextStyle(
                      fontSize: 15,
                      fontFeatures: [FontFeature.tabularFigures()],
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      }).toList(),
    );
  }

  String _value(List<double> values, int index, int digits) {
    if (values.length <= index) return "-";
    return values[index].toStringAsFixed(digits);
  }
}
