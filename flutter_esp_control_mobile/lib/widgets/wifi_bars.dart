import 'package:flutter/material.dart';

Widget wifiBars(int rssi) {
  int level = 1;

  if (rssi >= -50) {
    level = 4;
  } else if (rssi >= -60)
    level = 3;
  else if (rssi >= -70) level = 2;

  return Row(
    mainAxisSize: MainAxisSize.min,
    children: List.generate(4, (i) {
      return Icon(
        Icons.signal_cellular_alt,
        size: 16,
        color: (i < level) ? Colors.black87 : Colors.black26,
      );
    }),
  );
}
