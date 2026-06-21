import 'package:flutter/material.dart';

class UiLayout {
  // ✅ 所有頁面統一最大寬度
  static const double pageMaxWidth = 1200;

  // ✅ 所有頁面統一 padding
  static const EdgeInsets pagePadding = EdgeInsets.all(16);

  // ✅ Mobile breakpoint
  static const double mobileBreakpoint = 700;

  // ✅ 桌機左右間距
  static const double gap = 24;

  // ✅ 卡片間距
  static const double cardGap = 12;

  // ✅ 右側控制欄統一寬度（Camera / Servo）
  static const BoxConstraints sidePanelConstraints = BoxConstraints(
    minWidth: 360,
    maxWidth: 460,
  );

  // ✅ 卡片統一高度（盡量接近）
  static const double cardMinHeight = 140;

  // ✅ 按鈕統一高度
  static const double buttonHeight = 42;

  // ✅ 輸入框統一高度 (isDense + contentPadding 控制)
  static const EdgeInsets fieldPadding =
      EdgeInsets.symmetric(horizontal: 12, vertical: 10);
}
