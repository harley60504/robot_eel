import 'dart:io';
import 'dart:typed_data';

Future<void> writeBytesToPath(String path, Uint8List bytes) {
  return File(path).writeAsBytes(bytes, flush: true);
}
