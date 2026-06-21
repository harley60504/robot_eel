import 'package:flutter/material.dart';

import '../api/esp_api.dart';
import '../widgets/mode_switch.dart';
import '../widgets/servo_table.dart';
import '../widgets/servo_control_panel.dart';
import '../widgets/motion_param.dart';
import '../widgets/system_status.dart';
import '../ui/ui_layout.dart';

class ServoPage extends StatefulWidget {
  const ServoPage({super.key});

  @override
  State<ServoPage> createState() => _ServoPageState();
}

class _ServoPageState extends State<ServoPage> {
  int mode = -1;

  @override
  void initState() {
    super.initState();

    final cached = WsControlApi.lastCtrlParams;
    if (cached != null) {
      mode = cached['mode'] ?? -1;
    }

    WsControlApi.ctrlParamsNotifier.addListener(_onCtrlParams);
  }

  void _onCtrlParams() {
    final msg = WsControlApi.ctrlParamsNotifier.value;
    if (!mounted || msg == null) return;

    final newMode = msg['mode'] ?? -1;
    if (newMode != mode) {
      setState(() => mode = newMode);
    }
  }

  @override
  void dispose() {
    WsControlApi.ctrlParamsNotifier.removeListener(_onCtrlParams);
    super.dispose();
  }

  Widget buildRightPanel() {
    if (mode == 2) {
      return const ServoControlPanel(centerCalibration: true);
    }

    if (mode == 3) {
      return const ServoControlPanel();
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: const [
        MotionParam(),
        SizedBox(height: UiLayout.gap),
        SystemStatus(),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    final width = MediaQuery.of(context).size.width;
    final isMobile = width < UiLayout.mobileBreakpoint;

    return LayoutBuilder(
      builder: (context, constraints) {
        return Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: UiLayout.pageMaxWidth),
            child: SingleChildScrollView(
              padding: UiLayout.pagePadding,
              child: ConstrainedBox(
                constraints: BoxConstraints(
                  minHeight:
                      constraints.maxHeight - UiLayout.pagePadding.vertical,
                ),
                child: isMobile
                    ? Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          const ModeSwitch(),
                          const SizedBox(height: UiLayout.gap),
                          const ServoTable(),
                          const SizedBox(height: UiLayout.gap),
                          buildRightPanel(),
                        ],
                      )
                    : Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.stretch,
                              children: const [
                                ModeSwitch(),
                                SizedBox(height: UiLayout.gap),
                                ServoTable(),
                              ],
                            ),
                          ),
                          const SizedBox(width: UiLayout.gap),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.stretch,
                              children: [
                                buildRightPanel(),
                              ],
                            ),
                          ),
                        ],
                      ),
              ),
            ),
          ),
        );
      },
    );
  }
}
