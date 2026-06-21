import 'package:flutter/material.dart';
import '../api/esp_api.dart';
import '../ui/ui_card.dart';
import '../ui/ui_layout.dart';

class ModeSwitch extends StatefulWidget {
  final bool compact;
  final int? selectedMode;
  final ValueChanged<int>? onModeSelected;

  const ModeSwitch({
    super.key,
    this.compact = false,
    this.selectedMode,
    this.onModeSelected,
  });

  @override
  State<ModeSwitch> createState() => _ModeSwitchState();
}

class _ModeSwitchState extends State<ModeSwitch> {
  int mode = -1;

  @override
  void initState() {
    super.initState();

    final cached = WsControlApi.lastCtrlParams;
    if (cached != null) {
      mode = cached["mode"] ?? -1;
    }

    WsControlApi.ctrlParamsNotifier.addListener(_onCtrlParamsChanged);
  }

  void _onCtrlParamsChanged() {
    final msg = WsControlApi.ctrlParamsNotifier.value;
    if (!mounted || msg == null) return;

    final newMode = msg["mode"] ?? -1;
    if (newMode != mode) {
      setState(() => mode = newMode);
    }
  }

  @override
  void dispose() {
    WsControlApi.ctrlParamsNotifier.removeListener(_onCtrlParamsChanged);
    super.dispose();
  }

  void setMode(int m) {
    widget.onModeSelected?.call(m);
    if (m <= 3) {
      WsControlApi.setParam({"mode": m});
    }
  }

  Widget modeBtn(String name, int m) {
    final isSel = (widget.selectedMode ?? mode) == m;

    return SizedBox(
      height: UiLayout.buttonHeight,
      child: ElevatedButton(
        style: ElevatedButton.styleFrom(
          backgroundColor: isSel ? Colors.blue : null,
          padding: const EdgeInsets.symmetric(horizontal: 10),
          textStyle: const TextStyle(fontSize: 15, fontWeight: FontWeight.w700),
        ),
        onPressed: () => setMode(m),
        child: FittedBox(child: Text(name, softWrap: false)),
      ),
    );
  }

  String modeName(int currentMode) {
    switch (currentMode) {
      case 0:
        return "Sin";
      case 1:
        return "CPG";
      case 2:
        return "Offset";
      case 3:
        return "Angle";
      case 4:
        return "Python";
      default:
        return "-";
    }
  }

  @override
  Widget build(BuildContext context) {
    return UiCard(
      title: "模式切換",
      minHeight: widget.compact ? 138 : 170,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(child: modeBtn("Sin", 0)),
              const SizedBox(width: 8),
              Expanded(child: modeBtn("CPG", 1)),
              const SizedBox(width: 8),
              Expanded(child: modeBtn("Offset", 2)),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(child: modeBtn("Angle", 3)),
              const SizedBox(width: 8),
              Expanded(child: modeBtn("Python", 4)),
              const Spacer(),
            ],
          ),
          SizedBox(height: widget.compact ? 8 : 12),
          Text("目前選單：${modeName(widget.selectedMode ?? mode)}"),
        ],
      ),
    );
  }
}
