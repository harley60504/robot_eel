import 'package:flutter/material.dart';
import 'ui_layout.dart';

class ResponsiveRow extends StatelessWidget {
  final List<Widget> leftChildren;
  final Widget rightPanel;

  const ResponsiveRow({
    super.key,
    required this.leftChildren,
    required this.rightPanel,
  });

  @override
  Widget build(BuildContext context) {
    final width = MediaQuery.of(context).size.width;
    final isMobile = width < UiLayout.mobileBreakpoint;

    if (isMobile) {
      return ListView(
        children: [
          ..._withGap(leftChildren, gap: UiLayout.cardGap),
          const SizedBox(height: 12),
          rightPanel,
        ],
      );
    }

    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Expanded(
          child: Column(
            children: _withGap(leftChildren, gap: UiLayout.cardGap),
          ),
        ),
        const SizedBox(width: UiLayout.gap),
        ConstrainedBox(
          constraints: UiLayout.sidePanelConstraints,
          child: SizedBox(
            width: double.infinity,
            child: rightPanel,
          ),
        ),
      ],
    );
  }

  List<Widget> _withGap(List<Widget> children, {required double gap}) {
    final out = <Widget>[];
    for (int i = 0; i < children.length; i++) {
      out.add(children[i]);
      if (i != children.length - 1) {
        out.add(SizedBox(height: gap));
      }
    }
    return out;
  }
}
