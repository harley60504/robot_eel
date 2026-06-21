import 'dart:async';
import 'package:flutter/material.dart';
import '../api/esp_api.dart';
import '../ui/ui_card.dart';
import '../ui/ui_layout.dart';

class MotionParam extends StatefulWidget {
  final bool compact;
  final bool embedded;

  const MotionParam({
    super.key,
    this.compact = false,
    this.embedded = false,
  });

  @override
  State<MotionParam> createState() => _MotionParamState();
}

class _MotionParamState extends State<MotionParam> {
  final freqCtrl = TextEditingController();
  final ampCtrl = TextEditingController();
  final lamCtrl = TextEditingController();
  final lCtrl = TextEditingController();

  double freq = double.nan;
  double amp = double.nan;
  double lambda = double.nan;
  double length = double.nan;

  StreamSubscription? _sub;
  bool _didInitText = false;
  bool _editing = false;

  @override
  void initState() {
    super.initState();
    _applyCtrlParams(WsControlApi.lastCtrlParams);
    _sub = WsControlApi.stream().listen((msg) {
      if (!mounted || msg is! Map || msg["type"] != "ctrl_params") return;
      _applyCtrlParams(msg);
    });
  }

  void _applyCtrlParams(Map? msg) {
    if (msg == null) return;

    setState(() {
      freq = _num(msg["frequency"]);
      amp = _num(msg["Ajoint"]);
      lambda = _num(msg["lambda"]);
      length = _num(msg["L"]);
    });

    if (_editing) return;

    if (!_didInitText) {
      freqCtrl.text = _fmt(msg["frequency"]);
      ampCtrl.text = _fmt(msg["Ajoint"]);
      lamCtrl.text = _fmt(msg["lambda"]);
      lCtrl.text = _fmt(msg["L"]);
      _didInitText = true;
      return;
    }

    if (freqCtrl.text.isEmpty) freqCtrl.text = _fmt(msg["frequency"]);
    if (ampCtrl.text.isEmpty) ampCtrl.text = _fmt(msg["Ajoint"]);
    if (lamCtrl.text.isEmpty) lamCtrl.text = _fmt(msg["lambda"]);
    if (lCtrl.text.isEmpty) lCtrl.text = _fmt(msg["L"]);
  }

  String _fmt(dynamic v) {
    if (v == null) return "";
    final n = (v as num).toDouble();
    return n.toStringAsFixed(2);
  }

  double _num(dynamic v) => v is num ? v.toDouble() : double.nan;

  String _status(double v, {String unit = ""}) {
    if (v.isNaN) return "-";
    return "${v.toStringAsFixed(2)}$unit";
  }

  void setParam(String key, TextEditingController ctrl) {
    final v = double.tryParse(ctrl.text.trim());
    if (v == null) return;
    WsControlApi.setParam({key: v});
    setState(() => _editing = false);
  }

  @override
  void dispose() {
    _sub?.cancel();
    freqCtrl.dispose();
    ampCtrl.dispose();
    lamCtrl.dispose();
    lCtrl.dispose();
    super.dispose();
  }

  InputDecoration _fieldDeco() => const InputDecoration(
        border: OutlineInputBorder(),
        isDense: true,
        contentPadding: UiLayout.fieldPadding,
      );

  @override
  Widget build(BuildContext context) {
    final content = LayoutBuilder(
      builder: (context, constraints) {
        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                Expanded(
                  child: paramTile(
                    label: "\u983b\u7387 (Hz)",
                    value: _status(freq, unit: " Hz"),
                    ctrl: freqCtrl,
                    onSet: () => setParam("frequency", freqCtrl),
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: paramTile(
                    label: "\u632f\u5e45 (deg)",
                    value: _status(amp, unit: " deg"),
                    ctrl: ampCtrl,
                    onSet: () => setParam("Ajoint", ampCtrl),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 10),
            Row(
              children: [
                Expanded(
                  child: paramTile(
                    label: "lambda",
                    value: _status(lambda),
                    ctrl: lamCtrl,
                    onSet: () => setParam("lambda", lamCtrl),
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: paramTile(
                    label: "L",
                    value: _status(length),
                    ctrl: lCtrl,
                    onSet: () => setParam("L", lCtrl),
                  ),
                ),
              ],
            ),
          ],
        );
      },
    );

    if (widget.embedded) return content;

    return UiCard(
      title: "\u53c3\u6578\u8a2d\u5b9a",
      minHeight: widget.compact ? 188 : 240,
      child: content,
    );
  }

  Widget paramTile({
    required String label,
    required String value,
    required TextEditingController ctrl,
    required VoidCallback onSet,
  }) {
    return Container(
      constraints: const BoxConstraints(minHeight: 116),
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: const Color(0xFF0B0F14),
        border: Border.all(color: const Color(0xFF252B33)),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text(
            label,
            overflow: TextOverflow.ellipsis,
            style: const TextStyle(fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: 2),
          Text(
            value,
            overflow: TextOverflow.ellipsis,
            style: const TextStyle(color: Colors.white70, fontSize: 12),
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: TextField(
                  controller: ctrl,
                  keyboardType: TextInputType.number,
                  onTap: () => setState(() => _editing = true),
                  onChanged: (_) => setState(() => _editing = true),
                  onSubmitted: (_) => onSet(),
                  decoration: _fieldDeco(),
                ),
              ),
              const SizedBox(width: 8),
              SizedBox(
                width: 52,
                height: UiLayout.buttonHeight,
                child: ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    padding: EdgeInsets.zero,
                    textStyle: const TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  onPressed: onSet,
                  child: const Text("\u8a2d\u5b9a", softWrap: false),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
