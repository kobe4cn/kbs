# 🔍 项目类型自动检测指南

基于 Cursor Rust Rules 系统的项目分析和类型识别算法，帮助 Claude Code 自动判断项目特征并推荐合适的开发标准。

## 🏗️ 项目复杂度分层检测

### 1. 简单项目 (Single Crate) 检测特征

**文件结构特征：**
- 仅存在 `src/main.rs` 或 `src/lib.rs`
- 无 `Cargo.toml` 的 workspace 配置
- 源代码总行数 < 5,000 行
- 模块数量 ≤ 5 个

**Cargo.toml 特征：**
```toml
[package]
name = "simple_app"
# 无 [workspace] 段
# 依赖数量 < 15 个
```

**推荐应用：**
- 学习项目和原型
- 简单工具和脚本
- 单一功能应用

### 2. 中等复杂度 (Multi-Feature) 检测特征

**文件结构特征：**
- 存在多个功能模块目录
- 源代码总行数 5,000-20,000 行
- 包含专门的配置、错误处理模块
- 具有集成测试目录

**Cargo.toml 特征：**
```toml
[package]
name = "medium_app"

[dependencies]
# 依赖数量 15-50 个
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
# 包含特定功能依赖
```

**推荐应用：**
- Web 服务和 API
- CLI 应用工具
- 数据处理服务

### 3. 复杂项目 (Workspace) 检测特征

**文件结构特征：**
- 存在根级 `Cargo.toml` 包含 `[workspace]`
- 多个子包目录
- 源代码总行数 > 20,000 行
- 复杂的依赖关系图

**Cargo.toml 特征：**
```toml
[workspace]
members = [
    "core",
    "api",
    "worker",
    "shared"
]

[workspace.dependencies]
# 工作区级别的依赖管理
```

**推荐应用：**
- 微服务架构
- 大型系统和平台
- 多团队协作项目

## 🎯 功能特征检测算法

### Web 服务项目检测

**依赖检测模式：**
```rust
// 检测 Cargo.toml 中的关键依赖
fn detect_web_service(cargo_toml: &CargoToml) -> bool {
    cargo_toml.dependencies.contains_key("axum") ||
    cargo_toml.dependencies.contains_key("warp") ||
    cargo_toml.dependencies.contains_key("actix-web") ||
    cargo_toml.dependencies.contains_key("rocket")
}
```

**文件模式检测：**
- `src/routes/` - 路由定义目录
- `src/handlers/` - 请求处理器目录
- `src/middleware/` - 中间件目录
- `src/models/` - 数据模型目录

**配置特征检测：**
- 端口配置 (port, bind_address)
- HTTP 服务器设置
- API 版本配置

**推荐技术栈：**
- Axum 0.8+ + SQLx + OpenAPI (utoipa)
- 分层架构 + 依赖注入
- Prometheus + OpenTelemetry 可观测性

### CLI 应用项目检测

**依赖检测模式：**
```rust
fn detect_cli_application(cargo_toml: &CargoToml) -> bool {
    cargo_toml.dependencies.contains_key("clap") ||
    cargo_toml.dependencies.contains_key("structopt") ||
    cargo_toml.bin.is_some()
}
```

**文件模式检测：**
- `src/cli.rs` - 命令定义文件
- `src/commands/` - 子命令实现
- `examples/` - 使用示例目录
- `README.md` 包含使用说明

**配置特征检测：**
- 命令行参数定义
- 子命令结构
- 帮助文档配置

**推荐技术栈：**
- Clap 4.0+ with derive macros
- anyhow 错误处理 + 用户友好消息
- figment 配置管理 + 环境变量

### gRPC 服务项目检测

**依赖检测模式：**
```rust
fn detect_grpc_service(cargo_toml: &CargoToml) -> bool {
    (cargo_toml.dependencies.contains_key("tonic") &&
     cargo_toml.dependencies.contains_key("prost")) ||
    std::path::Path::new("proto").exists()
}
```

**文件模式检测：**
- `proto/` - Protocol Buffer 定义目录
- `build.rs` - protobuf 编译脚本
- `src/grpc/` - gRPC 服务实现

**配置特征检测：**
- protobuf 编译设置
- gRPC 服务端口配置
- 服务发现配置

**推荐技术栈：**
- Tonic 0.13+ + Prost
- Inner types + MessageSanitizer trait
- 分布式追踪 + 服务网格就绪

### 数据库项目检测

**依赖检测模式：**
```rust
fn detect_database_usage(cargo_toml: &CargoToml) -> bool {
    cargo_toml.dependencies.contains_key("sqlx") ||
    cargo_toml.dependencies.contains_key("diesel") ||
    cargo_toml.dependencies.contains_key("sea-orm")
}
```

**文件模式检测：**
- `migrations/` - 数据库迁移文件
- `src/models/` - 数据模型定义
- `src/repositories/` - 仓储模式实现
- `.env` 包含数据库连接配置

**配置特征检测：**
- 数据库连接字符串
- 连接池配置
- 迁移管理设置

**推荐技术栈：**
- SQLx (编译时检查，绝不使用 rusqlite)
- 仓储模式 + 事务管理
- 连接池 + 查询优化

## 🔧 并发和异步检测

**依赖检测模式：**
```rust
fn detect_async_usage(cargo_toml: &CargoToml) -> bool {
    cargo_toml.dependencies.contains_key("tokio") ||
    cargo_toml.dependencies.contains_key("async-std") ||
    source_contains_async_keywords()
}
```

**代码模式检测：**
- `async fn` 关键字使用频率
- `await` 关键字出现次数
- `Arc<DashMap>` 替代 `Arc<Mutex<HashMap>>`
- 消息传递 (channels) 使用模式

**推荐模式：**
- tokio 异步运行时
- DashMap 高性能并发集合
- 无锁数据结构优先

## 📊 项目健康度评估

### 代码质量指标
```rust
struct ProjectHealth {
    // 代码覆盖率
    test_coverage: f32,

    // Clippy 警告数量
    clippy_warnings: u32,

    // 依赖安全状况
    security_vulnerabilities: u32,

    // 性能基准存在性
    has_benchmarks: bool,

    // 文档完整度
    documentation_coverage: f32,
}
```

### 技术债务检测
- `unwrap()` 和 `expect()` 使用统计
- `unsafe` 代码块数量和文档
- 长函数 (>150行) 统计
- 循环复杂度分析

### 生产就绪度评估
- 错误处理覆盖率
- 日志和监控集成
- 容器化配置存在性
- CI/CD 管道配置

## 🤖 自动化检测实现

### 文件扫描器
```rust
pub struct ProjectAnalyzer {
    root_path: PathBuf,
    cargo_toml: CargoToml,
    file_stats: FileStats,
}

impl ProjectAnalyzer {
    pub fn analyze(&self) -> ProjectAnalysis {
        ProjectAnalysis {
            complexity: self.detect_complexity(),
            features: self.detect_features(),
            tech_stack: self.recommend_tech_stack(),
            architecture: self.recommend_architecture(),
            health_score: self.calculate_health_score(),
        }
    }
}
```

### 智能推荐引擎
基于检测结果提供：

1. **即时架构建议** - 根据项目特征推荐最佳架构模式
2. **技术栈优化** - 基于生产经验推荐依赖组合
3. **代码质量改进** - 识别技术债务和改进点
4. **性能优化建议** - 基于项目规模推荐优化策略
5. **安全加固方案** - 根据项目类型推荐安全实践

这个检测系统确保 Claude Code 能够快速、准确地识别项目特征，并提供最相关的开发指导和架构建议。
