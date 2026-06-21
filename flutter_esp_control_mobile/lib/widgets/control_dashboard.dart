import 'package:flutter/material.dart';
import '../api/esp_api.dart';
import '../config.dart';
import '../pages/python_page.dart';
import '../ui/ui_card.dart';
import '../ui/ui_layout.dart';
import 'camera_stream.dart';
import 'motion_param.dart';
import 'servo_control_panel.dart';

class ControlDashboard extends StatefulWidget {
  final int selectedMode;
  final ValueChanged<int> onModeSelected;
  final bool fillHeight;

  const ControlDashboard({
    super.key,
    required this.selectedMode,
    required this.onModeSelected,
    this.fillHeight = false,
  });

  @override
  State<ControlDashboard> createState() => _ControlDashboardState();
}

class _ControlDashboardState extends State<ControlDashboard> {
  static const modes = [
    (name: "Sin", value: 0),
    (name: "CPG", value: 1),
    (name: "Offset", value: 2),
    (name: "Angle", value: 3),
    (name: "Python", value: 4),
  ];

  void setMode(int mode) {
    widget.onModeSelected(mode);
    if (mode <= 3) {
      WsControlApi.setParam({"mode": mode});
    }
  }

  String modeName(int mode) {
    return modes
        .firstWhere((item) => item.value == mode, orElse: () => modes.first)
        .name;
  }

  Widget buildModeGrid() {
    Widget button(({String name, int value}) item) {
      final selected = widget.selectedMode == item.value;
      return SizedBox(
        height: UiLayout.buttonHeight,
        child: ElevatedButton(
          onPressed: () => setMode(item.value),
          style: ElevatedButton.styleFrom(
            backgroundColor: selected ? Colors.blue : null,
            textStyle: const TextStyle(fontWeight: FontWeight.w700),
            padding: const EdgeInsets.symmetric(horizontal: 8),
          ),
          child: FittedBox(child: Text(item.name)),
        ),
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Row(
          children: [
            Expanded(child: button(modes[0])),
            const SizedBox(width: 8),
            Expanded(child: button(modes[1])),
            const SizedBox(width: 8),
            Expanded(child: button(modes[2])),
          ],
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            Expanded(child: button(modes[3])),
            const SizedBox(width: 8),
            Expanded(child: button(modes[4])),
          ],
        ),
      ],
    );
  }

  Widget buildActiveControl() {
    if (widget.selectedMode == 4) {
      return PythonPage(
        compact: true,
        fillHeight: widget.fillHeight,
        embedded: true,
      );
    }
    if (widget.selectedMode == 3) {
      return const ServoControlPanel(compact: true, embedded: true);
    }
    if (widget.selectedMode == 2) {
      return const ServoControlPanel(
        compact: true,
        embedded: true,
        centerCalibration: true,
      );
    }
    return const MotionParam(compact: true, embedded: true);
  }

  Widget buildEsp32MiniPreview() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Text(
          "ESP32 Camera",
          style: Theme.of(context).textTheme.titleSmall,
        ),
        const SizedBox(height: 8),
        SizedBox(
          height: 118,
          child: ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: CameraStreamWS(
              wsUrl: ApiConfig.wsStreamUrl,
              initiallyPaused: true,
              showFullscreenButton: false,
            ),
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    final header = Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Text(
          "\u6a21\u5f0f\u5207\u63db",
          style: Theme.of(context).textTheme.titleMedium,
        ),
        const SizedBox(height: 10),
        buildModeGrid(),
        const SizedBox(height: 10),
        Text(
          "\u76ee\u524d\u9078\u55ae\uff1a${modeName(widget.selectedMode)}",
          style: const TextStyle(color: Colors.white70),
        ),
        const SizedBox(height: 16),
        const Divider(),
        const SizedBox(height: 12),
      ],
    );

    final esp32Preview = Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const Divider(),
        const SizedBox(height: 12),
        buildEsp32MiniPreview(),
      ],
    );

    final content = widget.fillHeight
        ? Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              header,
              Expanded(
                child: SingleChildScrollView(
                  child: buildActiveControl(),
                ),
              ),
              const SizedBox(height: 12),
              esp32Preview,
            ],
          )
        : Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              header,
              buildActiveControl(),
              const SizedBox(height: 16),
              esp32Preview,
            ],
          );

    return UiCard(
      title: "\u5100\u9336\u677f",
      minHeight: widget.fillHeight ? 0 : 520,
      fill: widget.fillHeight,
      child: content,
    );
  }
}
