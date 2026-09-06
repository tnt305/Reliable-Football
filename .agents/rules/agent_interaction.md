---
trigger: always_on
---

# Agent Interaction Rules

## Thông báo khi kích hoạt Skill

- **BẮT BUỘC**: Bất cứ khi nào Agent tham chiếu, kích hoạt hoặc sử dụng một Skill từ `.agents/skills/` (như `analyze-results`, `train-soccernet`, `audio-pipeline`, `dataset-debug`, `soccernet-download`, `add-model-variant`):
  - Agent **PHẢI** hiển thị thông báo rõ ràng ở đầu phản hồi để User nắm được:
    `🛠️ [Kích hoạt Skill: <tên-skill>] - <Mục đích sử dụng>`
  - Báo cáo rõ ràng các bước mà skill hướng dẫn đang được thực hiện.
