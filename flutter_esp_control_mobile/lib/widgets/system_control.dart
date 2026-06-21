import 'package:flutter/material.dart';

import '../ui/ui_card.dart';

class SystemControl extends StatefulWidget {
  const SystemControl({super.key});

  @override
  State<SystemControl> createState() => _SystemControlState();
}

class _SystemControlState extends State<SystemControl> {
  @override
  Widget build(BuildContext context) {
    return UiCard(
      title: "System Control",
      child: Padding(
        padding: EdgeInsets.zero,
        child: Row(
          children: [
            ElevatedButton(
                onPressed: () {}, child: const Text("Pause / Resume")),
          ],
        ),
      ),
    );
  }
}
