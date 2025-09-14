# Repository Guidelines

## 项目结构与模块组织
- `src/` — Rust 库 crate（包名 `kbs`）。单元测试可内联；集成测试放 `tests/`。
- `ui/` — Vite + React + TypeScript。组件在 `ui/src/components/ui/`；静态资源在 `ui/public/`。
- `examples/`, `fixtures/` — 示例与测试数据；`specs/` 存放 AI 规范；`rules/` 为内部知识库。
- `_github/workflows/` — CI。根目录 `Makefile` 提供常用任务；元数据见 `Cargo.toml`。

## 构建、测试与本地开发
- Rust 构建：`make build`（等价 `cargo build`）。
- Rust 测试：`make test`（Nextest，含全部特性）。
- 格式/静态检查：`cargo fmt --all`；`cargo clippy --all-targets --all-features --tests --benches -- -D warnings`。
- 依赖与质量：`cargo deny check -d`；拼写检查 `typos`；覆盖率：`cargo llvm-cov --summary-only`。
- UI 开发：`cd ui && yarn dev`（启动 Vite dev server）。
- UI 构建/测试：`yarn build`；`yarn test`（Vitest）。

## 代码风格与命名
- Rust：rustfmt（4 空格缩进）。模块 `snake_case.rs`；类型 `PascalCase`；函数/变量 `snake_case`；常量 `SCREAMING_SNAKE_CASE`。
- 错误处理：库内优先 `thiserror`；二进制使用 `anyhow::Result`；避免在库中使用 `.unwrap()`。
- TS/React：规则见 `ui/eslint.config.js`。文件用 kebab-case，导出 `PascalCase` 组件（如 `button.tsx` 导出 `Button`）。组件用 `tsx`，工具用 `ts`。JSX 可使用 Tailwind 类名。

## 测试指南
- Rust：单元测试置于 `#[cfg(test)]` 模块；集成测试在 `tests/`；需要样例请用 `fixtures/`。运行：`make test`。
- UI：测试文件 `*.test.ts`/`*.test.tsx`；使用 Vitest 与 JSDOM 模拟；保持测试确定性。

## Commit 与 Pull Request
- 使用 Conventional Commits：`feat`/`fix`/`doc`/`perf`/`refactor`/`style`/`test`/`chore`。示例：`feat(ui): add sidebar accordion`。
- PR 要求：清晰描述、关联问题（如 `#123`）、UI 变更附截图，必要时引用 `specs/` 与 `rules/`。确保格式化、Lint、测试均通过。
- 提交前检查：`pre-commit install && pre-commit run --all-files`。

## 安全与配置
- 禁止提交密钥；`.env` 保持本地（参见 `ui/.env`）。定期运行 `cargo deny`。CI 目标分支为 `main`；发布从提交打标签。

## Agent 提示
- 变更聚焦且最小化；勿顺带修复无关问题。大型变更先开 Issue 协调。
