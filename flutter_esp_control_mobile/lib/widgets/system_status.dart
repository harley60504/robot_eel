import 'dart:async';
import 'package:flutter/material.dart';
import '../api/esp_api.dart';
import '../ui/ui_card.dart';

class SystemStatus extends StatefulWidget {
  final bool compact;

  const SystemStatus({super.key, this.compact = false});

  @override
  State<SystemStatus> createState() => _SystemStatusState();
}

class _SystemStatusState extends State<SystemStatus> {
  double freq = double.nan;
  double amp = double.nan;
  double lambda = double.nan;
  double L = double.nan;
  double fbGain = double.nan;
  bool paused = false;

  StreamSubscription? _sub;

  @override
  void initState() {
    super.initState();

    _apply(WsControlApi.lastCtrlParams);

    _sub = WsControlApi.stream().listen((msg) {
      if (!mounted) return;
      if (msg is! Map) return;
      if (msg["type"] != "ctrl_params") return;
      _apply(msg);
    });
  }

  void _apply(Map? msg) {
    if (msg == null) return;

    setState(() {
      freq = (msg["frequency"] ?? double.nan).toDouble();
      amp = (msg["Ajoint"] ?? double.nan).toDouble();
      lambda = (msg["lambda"] ?? double.nan).toDouble();
      L = (msg["L"] ?? double.nan).toDouble();
      fbGain =
          (msg["feedbackGain"] ?? msg["feedback"] ?? double.nan).toDouble();
      paused = (msg["paused"] ?? false);
    });
  }

  @override
  void dispose() {
    _sub?.cancel();
    super.dispose();
  }

  String fmt(double v, {String unit = ""}) {
    return v.isNaN ? "-" : "${v.toStringAsFixed(2)}$unit";
  }

  @override
  Widget build(BuildContext context) {
    final summary = Wrap(
      spacing: 12,
      runSpacing: 6,
      children: [
        Text("頻率 ${fmt(freq, unit: " Hz")}"),
        Text("振幅 ${fmt(amp, unit: " °")}"),
        Text("λ ${fmt(lambda)}"),
        Text("L ${fmt(L)}"),
        Text("回授 ${fmt(fbGain)}"),
        Text(paused ? "暫停" : "運行中"),
      ],
    );

    if (widget.compact) {
      return summary;
    }

    return UiCard(
      title: "系統狀態",
      minHeight: 160,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [summary],
      ),
    );
  }
}
