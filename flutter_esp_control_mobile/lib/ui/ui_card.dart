import 'package:flutter/material.dart';
import 'ui_layout.dart';

class UiCard extends StatelessWidget {
  static const double padding = 16;
  static const double titleGap = 20;

  final String title;
  final Widget child;

  /// ✅ 統一卡片最小高度（用來讓左右看起來不要差太多）
  final double minHeight;
  final bool fill;

  const UiCard({
    super.key,
    required this.title,
    required this.child,
    this.minHeight = UiLayout.cardMinHeight,
    this.fill = false,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      child: Card(
        elevation: 3,
        child: Padding(
          padding: const EdgeInsets.all(padding),
          child: fill
              ? Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(title, style: const TextStyle(fontSize: 20)),
                    const SizedBox(height: titleGap),
                    Expanded(child: child),
                  ],
                )
              : ConstrainedBox(
                  constraints: BoxConstraints(minHeight: minHeight),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(title, style: const TextStyle(fontSize: 20)),
                      const SizedBox(height: titleGap),
                      child,
                    ],
                  ),
                ),
        ),
      ),
    );
  }
}
