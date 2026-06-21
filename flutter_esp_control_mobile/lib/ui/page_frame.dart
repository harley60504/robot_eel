import 'package:flutter/material.dart';
import 'ui_layout.dart';

class PageFrame extends StatelessWidget {
  final Widget child;

  const PageFrame({super.key, required this.child});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: UiLayout.pageMaxWidth),
        child: Padding(
          padding: UiLayout.pagePadding,
          child: child,
        ),
      ),
    );
  }
}
