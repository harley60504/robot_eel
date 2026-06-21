import 'package:flutter/material.dart';
import '../widgets/camera_stream.dart';
import '../widgets/camera_control.dart';
import '../config.dart';
import '../ui/ui_layout.dart';

class CameraPage extends StatelessWidget {
  const CameraPage({super.key});

  @override
  Widget build(BuildContext context) {
    final width = MediaQuery.of(context).size.width;
    final isMobile = width < UiLayout.mobileBreakpoint;

    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: UiLayout.pageMaxWidth),
        child: Padding(
          padding: UiLayout.pagePadding,
          child: isMobile
              ? ListView(
                  children: [
                    AspectRatio(
                      aspectRatio: 4 / 3,
                      child: CameraStreamWS(wsUrl: ApiConfig.wsStreamUrl),
                    ),
                    const SizedBox(height: UiLayout.gap),
                    const CameraControlPanel(),
                  ],
                )
              : Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Expanded(
                      child: AspectRatio(
                        aspectRatio: 4 / 3,
                        child: CameraStreamWS(wsUrl: ApiConfig.wsStreamUrl),
                      ),
                    ),
                    const SizedBox(width: UiLayout.gap),
                    ConstrainedBox(
                      constraints: UiLayout.sidePanelConstraints,
                      child: const SizedBox(
                        width: double.infinity, // ✅ 右側寬度統一
                        child: CameraControlPanel(),
                      ),
                    ),
                  ],
                ),
        ),
      ),
    );
  }
}
