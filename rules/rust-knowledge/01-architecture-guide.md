# 🏛️ 架构选择指南

基于 Cursor Rust Rules 系统提炼的架构决策指南，帮助 Claude Code 为不同类型的 Rust 项目推荐最佳架构模式和技术栈组合。

## 🎯 架构决策矩阵

### 项目类型 × 架构模式映射

| 项目类型 | 简单架构 | 中等架构 | 复杂架构 |
|---------|---------|---------|---------|
| **Web 服务** | Axum + SQLx | 分层架构 + OpenAPI | 微服务 + 事件驱动 |
| **CLI 工具** | Clap + 简单结构 | 子命令 + 插件系统 | 复杂工作流引擎 |
| **gRPC 服务** | 单服务 + Tonic | 多服务 + 负载均衡 | 服务网格 + 注册发现 |
| **数据处理** | 单线程批处理 | 并发流处理 | 分布式计算 |
| **系统工具** | 直接系统调用 | 异步 I/O + 池 | 高性能运行时 |

## 🌐 Web 服务架构模式

### 1. 简单 Web 服务架构

**适用场景：**
- API 服务 < 10 个端点
- 单一数据源
- 团队规模 ≤ 3 人

**推荐技术栈：**
```toml
[dependencies]
axum = "0.8"
tokio = { version = "1.0", features = ["full"] }
sqlx = { version = "0.8", features = ["runtime-tokio-rustls", "postgres"] }
serde = { version = "1.0", features = ["derive"] }
anyhow = "1.0"
```

**架构模式：**
```
src/
├── main.rs              # 最小启动入口
├── lib.rs               # 核心应用逻辑
├── routes.rs            # 路由定义
├── handlers.rs          # 请求处理器
├── models.rs            # 数据模型
├── database.rs          # 数据库连接
└── errors.rs            # 错误定义
```

**典型实现模板：**
```rust
// src/lib.rs
use axum::{Router, routing::get};
use sqlx::PgPool;

pub struct AppState {
    db: PgPool,
}

pub fn create_app(db: PgPool) -> Router {
    let state = AppState { db };

    Router::new()
        .route("/health", get(health_check))
        .route("/api/users", get(list_users).post(create_user))
        .with_state(state)
}
```

### 2. 中等复杂度 Web 架构

**适用场景：**
- API 服务 10-50 个端点
- 多数据源集成
- 需要认证和授权
- 团队规模 3-8 人

**推荐技术栈：**
```toml
[dependencies]
axum = "0.8"
utoipa = "4.0"                    # OpenAPI 文档
utoipa-swagger-ui = "4.0"
tower = "0.4"                     # 中间件
tower-http = { version = "0.5", features = ["cors", "trace"] }
jsonwebtoken = "9.0"              # JWT 认证
figment = { version = "0.10", features = ["toml", "env"] }
```

**分层架构模式：**
```
src/
├── main.rs                       # 启动入口
├── lib.rs                        # 应用配置
├── config.rs                     # 配置管理
├── routes/                       # 路由层
│   ├── mod.rs
│   ├── auth.rs
│   └── users.rs
├── handlers/                     # 处理器层
│   ├── mod.rs
│   ├── auth.rs
│   └── users.rs
├── services/                     # 业务逻辑层
│   ├── mod.rs
│   ├── auth_service.rs
│   └── user_service.rs
├── repositories/                 # 数据访问层
│   ├── mod.rs
│   ├── user_repository.rs
│   └── traits.rs
├── models/                       # 数据模型
│   ├── mod.rs
│   ├── user.rs
│   └── auth.rs
├── middleware/                   # 中间件
│   ├── mod.rs
│   ├── auth.rs
│   └── logging.rs
└── errors.rs                     # 错误处理
```

**依赖注入模式：**
```rust
// src/lib.rs
#[derive(Clone)]
pub struct AppState {
    pub user_service: Arc<UserService>,
    pub auth_service: Arc<AuthService>,
    pub config: Arc<Config>,
}

pub async fn create_app(config: Config) -> anyhow::Result<Router> {
    let db_pool = create_db_pool(&config.database_url).await?;

    let user_repository = Arc::new(PgUserRepository::new(db_pool.clone()));
    let user_service = Arc::new(UserService::new(user_repository));
    let auth_service = Arc::new(AuthService::new(config.jwt_secret.clone()));

    let state = AppState {
        user_service,
        auth_service,
        config: Arc::new(config),
    };

    Ok(create_router(state))
}
```

### 3. 复杂 Web 服务架构

**适用场景：**
- 微服务架构
- 高并发场景 (>10k QPS)
- 多团队协作
- 需要水平扩展

**推荐技术栈：**
```toml
[dependencies]
# 核心框架
axum = "0.8"
tokio = { version = "1.0", features = ["full"] }

# 微服务支持
tonic = "0.13"                    # gRPC 服务间通信
consul = "0.4"                    # 服务注册发现
redis = "0.25"                    # 分布式缓存

# 事件驱动
lapin = "2.0"                     # RabbitMQ 客户端
kafka = "0.9"                     # Apache Kafka

# 可观测性
tracing = "0.1"
tracing-opentelemetry = "0.22"
prometheus = "0.13"
```

**微服务架构模式：**
```
project/
├── Cargo.toml                    # 工作区配置
├── shared/                       # 共享库
│   ├── models/
│   ├── errors/
│   └── utils/
├── api-gateway/                  # API 网关
│   ├── src/
│   └── Cargo.toml
├── user-service/                 # 用户服务
│   ├── src/
│   └── Cargo.toml
├── auth-service/                 # 认证服务
│   ├── src/
│   └── Cargo.toml
├── notification-service/         # 通知服务
│   ├── src/
│   └── Cargo.toml
└── deployment/                   # 部署配置
    ├── docker/
    ├── k8s/
    └── helm/
```

## 🖥️ CLI 应用架构模式

### 1. 简单 CLI 架构

**适用场景：**
- 单一主要功能
- 配置选项 < 20 个
- 无插件需求

**推荐技术栈：**
```toml
[dependencies]
clap = { version = "4.0", features = ["derive"] }
anyhow = "1.0"
serde = { version = "1.0", features = ["derive"] }
toml = "0.8"
```

**架构模式：**
```rust
// src/main.rs
use clap::Parser;

#[derive(Parser)]
#[command(name = "my-tool")]
#[command(about = "A simple CLI tool")]
struct Cli {
    #[arg(short, long)]
    input: String,

    #[arg(short, long, default_value = "output.txt")]
    output: String,

    #[arg(short, long)]
    verbose: bool,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    my_tool::run(cli)
}
```

### 2. 复杂 CLI 架构

**适用场景：**
- 多子命令结构
- 插件系统支持
- 复杂配置管理

**推荐技术栈：**
```toml
[dependencies]
clap = { version = "4.0", features = ["derive"] }
anyhow = "1.0"
figment = { version = "0.10", features = ["toml", "yaml", "env"] }
libloading = "0.8"                # 动态插件加载
dashmap = "5.0"                   # 并发安全集合
```

**架构模式：**
```
src/
├── main.rs                       # 入口点
├── lib.rs                        # 核心逻辑
├── cli.rs                        # 命令定义
├── config.rs                     # 配置管理
├── commands/                     # 子命令实现
│   ├── mod.rs
│   ├── init.rs
│   ├── build.rs
│   └── deploy.rs
├── plugins/                      # 插件系统
│   ├── mod.rs
│   ├── loader.rs
│   └── registry.rs
├── utils/                        # 工具函数
│   ├── mod.rs
│   ├── fs.rs
│   └── git.rs
└── errors.rs                     # 错误处理
```

## 🚀 gRPC 服务架构模式

### 1. 单服务 gRPC 架构

**技术栈：**
```toml
[dependencies]
tonic = "0.13"
prost = "0.13"
tokio = { version = "1.0", features = ["full"] }

[build-dependencies]
tonic-build = "0.13"
```

**架构模式：**
```
src/
├── main.rs                       # 服务启动
├── lib.rs                        # 核心逻辑
├── grpc/                         # gRPC 服务实现
│   ├── mod.rs
│   └── user_service.rs
├── models/                       # 内部数据模型
│   ├── mod.rs
│   └── user.rs
├── repositories/                 # 数据访问
└── proto/                        # Protocol Buffer 定义
    └── user.proto
```

**Inner Types 模式：**
```rust
// src/models/user.rs
#[derive(Debug, Clone)]
pub struct User {
    pub id: uuid::Uuid,
    pub email: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

// 转换实现
impl From<User> for proto::User {
    fn from(user: User) -> Self {
        Self {
            id: user.id.to_string(),
            email: user.email,
            created_at: Some(user.created_at.into()),
        }
    }
}
```

### 2. 微服务 gRPC 架构

**服务网格就绪架构：**
```
services/
├── Cargo.toml                    # 工作区配置
├── proto/                        # 共享 Protocol Buffers
│   ├── user.proto
│   ├── auth.proto
│   └── notification.proto
├── shared/                       # 共享库
│   ├── models/
│   ├── middleware/
│   └── observability/
├── user-service/                 # 用户服务
├── auth-service/                 # 认证服务
├── api-gateway/                  # gRPC-HTTP 网关
└── deployment/
    ├── envoy/                    # 服务网格配置
    ├── kubernetes/
    └── helm/
```

## 📊 数据处理架构模式

### 流处理架构

**实时数据处理：**
```toml
[dependencies]
tokio = { version = "1.0", features = ["full"] }
tokio-stream = "0.1"
futures = "0.3"
dashmap = "5.0"                   # 高性能并发 Map
```

**架构模式：**
```rust
use tokio_stream::{Stream, StreamExt};
use dashmap::DashMap;
use std::sync::Arc;

pub struct StreamProcessor<T> {
    state: Arc<DashMap<String, T>>,
    config: ProcessorConfig,
}

impl<T> StreamProcessor<T> {
    pub async fn process_stream<S>(&self, stream: S)
    where
        S: Stream<Item = T>,
    {
        stream
            .for_each_concurrent(self.config.concurrency, |item| {
                let state = self.state.clone();
                async move {
                    // 处理逻辑
                }
            })
            .await;
    }
}
```

## 🔧 性能优化架构模式

### 高性能服务架构

**零成本抽象 + 无锁并发：**
```rust
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use arc_swap::ArcSwap;

pub struct HighPerformanceService {
    // 无锁计数器
    request_counter: AtomicU64,

    // 无锁并发 Map
    cache: DashMap<String, CachedData>,

    // 原子配置热更新
    config: ArcSwap<ServiceConfig>,
}

impl HighPerformanceService {
    pub async fn handle_request(&self, request: Request) -> Response {
        // 原子递增计数
        self.request_counter.fetch_add(1, Ordering::Relaxed);

        // 无锁缓存访问
        if let Some(cached) = self.cache.get(&request.key) {
            return cached.clone();
        }

        // 处理请求
        self.process_request(request).await
    }
}
```

## 🛡️ 安全架构模式

### 安全优先设计

**多层安全架构：**
```
Security Layers:
├── Network Security              # TLS, mTLS, VPN
├── Authentication               # JWT, OAuth2, mTLS certs
├── Authorization                # RBAC, ABAC policies
├── Data Protection             # Encryption at rest/transit
├── Input Validation            # Schema validation, sanitization
├── Audit Logging               # Security events tracking
└── Runtime Protection          # Rate limiting, DDoS protection
```

**安全组件集成：**
```toml
[dependencies]
# 认证和加密
jsonwebtoken = "9.0"
argon2 = "0.5"                    # 密码哈希
ring = "0.17"                     # 加密原语
rustls = "0.23"                   # TLS 实现

# 输入验证
validator = "0.16"
serde_json = "1.0"
```

这个架构指南确保 Claude Code 能够根据项目需求和复杂度，推荐最适合的架构模式和技术栈组合，同时遵循生产级质量标准。
