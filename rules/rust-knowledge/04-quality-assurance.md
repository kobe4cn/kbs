# 🛡️ 质量保证体系

基于 Cursor Rust Rules 质量层规则整理的完整质量保证体系，为 Claude Code 提供全面的代码质量检查、测试策略和生产就绪验证标准。

## 🚨 错误处理质量标准

### 1. 错误处理策略分层

**库项目错误处理 (thiserror)：**
```rust
// src/errors.rs - 库项目错误定义
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MyLibError {
    #[error("Invalid input: {message}")]
    InvalidInput { message: String },

    #[error("File not found: {path}")]
    FileNotFound { path: String },

    #[error("Permission denied")]
    PermissionDenied,

    #[error("Network error")]
    Network(#[from] reqwest::Error),

    #[error("IO error")]
    Io(#[from] std::io::Error),

    #[error("Parse error: {0}")]
    Parse(#[from] serde_json::Error),

    #[error("Configuration error: {message}")]
    Config { message: String },

    #[error("Internal error: {0}")]
    Internal(String),
}

// 结果类型别名
pub type Result<T> = std::result::Result<T, MyLibError>;

// 错误构造助手
impl MyLibError {
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput {
            message: message.into(),
        }
    }

    pub fn file_not_found(path: impl Into<String>) -> Self {
        Self::FileNotFound {
            path: path.into(),
        }
    }

    pub fn config(message: impl Into<String>) -> Self {
        Self::Config {
            message: message.into(),
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
    }
}

// 在库函数中使用
pub fn process_file(path: &str) -> Result<String> {
    if path.is_empty() {
        return Err(MyLibError::invalid_input("Path cannot be empty"));
    }

    let content = std::fs::read_to_string(path)
        .map_err(|_| MyLibError::file_not_found(path))?;

    if content.is_empty() {
        return Err(MyLibError::invalid_input("File is empty"));
    }

    Ok(content.to_uppercase())
}
```

**二进制项目错误处理 (anyhow)：**
```rust
// src/errors.rs - 二进制项目错误定义
use anyhow::{Context, Result};
use thiserror::Error;

// 应用特定的结构化错误
#[derive(Error, Debug)]
pub enum AppError {
    #[error("Configuration error: {message}")]
    Config { message: String },

    #[error("User error: {message}")]
    User { message: String },

    #[error("System error: {message}")]
    System { message: String },
}

// 错误构造助手
pub fn config_error(message: impl Into<String>) -> AppError {
    AppError::Config {
        message: message.into(),
    }
}

pub fn user_error(message: impl Into<String>) -> AppError {
    AppError::User {
        message: message.into(),
    }
}

// 主函数中的错误处理
fn main() -> Result<()> {
    let config = load_config()
        .context("Failed to load application configuration")?;

    let result = process_data(&config)
        .context("Failed to process data")?;

    save_results(&result)
        .context("Failed to save results")?;

    println!("Processing completed successfully");
    Ok(())
}

fn load_config() -> Result<Config> {
    let config_path = std::env::var("CONFIG_PATH")
        .context("CONFIG_PATH environment variable not set")?;

    anyhow::ensure!(!config_path.is_empty(), config_error("Config path is empty"));

    let content = std::fs::read_to_string(&config_path)
        .with_context(|| format!("Failed to read config file: {}", config_path))?;

    let config: Config = toml::from_str(&content)
        .context("Failed to parse config file as TOML")?;

    Ok(config)
}
```

### 2. 错误处理反模式检测

**自动化反模式识别：**
```rust
// ❌ 绝不允许的模式
fn bad_error_handling() {
    // 1. unwrap() 在生产代码中
    let value = risky_operation().unwrap();  // 检测并报告

    // 2. 忽略错误
    let _ = might_fail();  // 检测并报告

    // 3. 通用错误消息
    return Err("Something went wrong".into());  // 检测并报告

    // 4. 不提供上下文
    let data = load_file(path)?; // 缺少上下文信息
}

// ✅ 正确的错误处理模式
fn good_error_handling() -> anyhow::Result<()> {
    // 1. 适当的错误处理和上下文
    let value = risky_operation()
        .context("Failed to perform risky operation")?;

    // 2. 显式处理或传播错误
    if let Err(e) = might_fail() {
        tracing::error!("Operation failed: {}", e);
        return Err(e.into());
    }

    // 3. 具体的错误消息
    anyhow::ensure!(!input.is_empty(), "Input cannot be empty");

    // 4. 丰富的上下文用于调试
    let data = load_file(path)
        .with_context(|| format!("Failed to load config file: {}", path))?;

    Ok(())
}
```

## 🧪 测试质量标准

### 1. 分层测试策略

**单元测试模式：**
```rust
// src/services/user_service.rs - 服务层测试
#[cfg(test)]
mod tests {
    use super::*;
    use mockall::predicate::*;
    use mockall::mock;

    // Mock 仓储用于单元测试
    mock! {
        UserRepository {}

        #[async_trait]
        impl Repository<User, UserId> for UserRepository {
            type Error = sqlx::Error;

            async fn create(&self, user: User) -> Result<User, Self::Error>;
            async fn find_by_id(&self, id: UserId) -> Result<Option<User>, Self::Error>;
            async fn update(&self, user: User) -> Result<User, Self::Error>;
            async fn delete(&self, id: UserId) -> Result<(), Self::Error>;
        }
    }

    #[tokio::test]
    async fn test_create_user_success() {
        // Arrange
        let mut mock_repo = MockUserRepository::new();
        let test_user = User {
            id: UserId::new(),
            email: "test@example.com".to_string(),
            name: "Test User".to_string(),
            created_at: chrono::Utc::now(),
        };

        mock_repo
            .expect_create()
            .with(eq(test_user.clone()))
            .times(1)
            .returning(|user| Ok(user));

        let service = UserService::new(Arc::new(mock_repo));

        // Act
        let result = service.create_user(test_user.clone()).await;

        // Assert
        assert!(result.is_ok());
        assert_eq!(result.unwrap().email, test_user.email);
    }

    #[tokio::test]
    async fn test_create_user_duplicate_email() {
        let mut mock_repo = MockUserRepository::new();

        mock_repo
            .expect_create()
            .returning(|_| Err(sqlx::Error::Database(
                Box::new(sqlx::postgres::PgDatabaseError::new(
                    "23505", // unique_violation
                    "duplicate key value violates unique constraint"
                ))
            )));

        let service = UserService::new(Arc::new(mock_repo));
        let test_user = User::new("test@example.com".to_string(), "Test".to_string());

        let result = service.create_user(test_user).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), UserServiceError::DuplicateEmail));
    }

    // 性能测试
    #[tokio::test]
    async fn test_create_user_performance() {
        let start = std::time::Instant::now();

        // 测试逻辑...

        let duration = start.elapsed();
        assert!(duration < std::time::Duration::from_millis(100),
               "User creation took too long: {:?}", duration);
    }
}
```

**集成测试模式：**
```rust
// tests/integration_test.rs - 集成测试
use sqlx::PgPool;
use testcontainers::{clients, images, Container};
use my_app::{create_app, AppConfig};

struct TestContext {
    db_pool: PgPool,
    _db_container: Container<'static, images::postgres::Postgres>,
}

impl TestContext {
    async fn new() -> Self {
        // 启动测试数据库容器
        let docker = clients::Cli::default();
        let db_container = docker.run(images::postgres::Postgres::default());

        let connection_string = format!(
            "postgres://postgres:postgres@127.0.0.1:{}/postgres",
            db_container.get_host_port_ipv4(5432)
        );

        let db_pool = PgPool::connect(&connection_string)
            .await
            .expect("Failed to connect to test database");

        // 运行迁移
        sqlx::migrate!("./migrations")
            .run(&db_pool)
            .await
            .expect("Failed to run migrations");

        Self {
            db_pool,
            _db_container: db_container,
        }
    }
}

#[tokio::test]
async fn test_user_crud_operations() {
    let ctx = TestContext::new().await;

    // 创建用户
    let create_request = CreateUserRequest {
        email: "test@example.com".to_string(),
        name: "Test User".to_string(),
    };

    let created_user = sqlx::query_as!(
        User,
        "INSERT INTO users (email, name) VALUES ($1, $2) RETURNING *",
        create_request.email,
        create_request.name
    )
    .fetch_one(&ctx.db_pool)
    .await
    .unwrap();

    // 验证创建
    assert_eq!(created_user.email, create_request.email);
    assert_eq!(created_user.name, create_request.name);

    // 查找用户
    let found_user = sqlx::query_as!(
        User,
        "SELECT * FROM users WHERE id = $1",
        created_user.id
    )
    .fetch_optional(&ctx.db_pool)
    .await
    .unwrap();

    assert!(found_user.is_some());
    assert_eq!(found_user.unwrap().id, created_user.id);
}

#[tokio::test]
async fn test_api_endpoints() {
    let ctx = TestContext::new().await;
    let config = AppConfig::test_config();
    let app = create_app(ctx.db_pool.clone(), config);

    // 使用 axum-test 进行 API 测试
    let client = axum_test::TestServer::new(app).unwrap();

    // 测试健康检查
    let response = client.get("/health").await;
    assert_eq!(response.status_code(), 200);

    // 测试创建用户
    let create_request = CreateUserRequest {
        email: "api_test@example.com".to_string(),
        name: "API Test User".to_string(),
    };

    let response = client
        .post("/api/v1/users")
        .json(&create_request)
        .await;

    assert_eq!(response.status_code(), 201);

    let user: User = response.json();
    assert_eq!(user.email, create_request.email);
}
```

### 2. 属性测试 (Property-Based Testing)

**使用 proptest 进行属性测试：**
```rust
// tests/property_tests.rs
use proptest::prelude::*;
use my_app::utils::{validate_email, normalize_string};

proptest! {
    #[test]
    fn test_email_validation_properties(
        local in "[a-zA-Z0-9]{1,20}",
        domain in "[a-zA-Z0-9]{1,10}",
        tld in "[a-zA-Z]{2,4}"
    ) {
        let email = format!("{}@{}.{}", local, domain, tld);
        prop_assert!(validate_email(&email));
    }

    #[test]
    fn test_string_normalization_idempotent(s in ".*") {
        let normalized = normalize_string(&s);
        let double_normalized = normalize_string(&normalized);
        prop_assert_eq!(normalized, double_normalized);
    }

    #[test]
    fn test_user_id_roundtrip(id in prop::collection::vec(prop::num::u8::ANY, 16)) {
        let uuid_bytes: [u8; 16] = id.try_into().unwrap();
        let user_id = UserId::from_bytes(uuid_bytes);
        let recovered_bytes = user_id.to_bytes();
        prop_assert_eq!(uuid_bytes, recovered_bytes);
    }
}

// 基准测试
#[cfg(test)]
mod benches {
    use super::*;
    use criterion::{black_box, criterion_group, criterion_main, Criterion};

    fn benchmark_user_creation(c: &mut Criterion) {
        let rt = tokio::runtime::Runtime::new().unwrap();

        c.bench_function("user_creation", |b| {
            b.to_async(&rt).iter(|| async {
                let user = User::new(
                    black_box("test@example.com".to_string()),
                    black_box("Test User".to_string()),
                );
                black_box(user)
            })
        });
    }

    criterion_group!(benches, benchmark_user_creation);
    criterion_main!(benches);
}
```

## 📊 代码质量监控

### 1. 静态分析配置

**Clippy 配置 (clippy.toml)：**
```toml
# 严格的 Clippy 规则
warn-on-all-wildcard-imports = true
disallowed-methods = [
    # 禁止使用危险方法
    { path = "std::unwrap", reason = "Use proper error handling" },
    { path = "std::expect", reason = "Use proper error handling in production" },
    { path = "std::panic", reason = "Use Result instead of panicking" },
    { path = "std::println", reason = "Use proper logging instead" },
]

disallowed-types = [
    # 禁止使用低效类型组合
    { path = "std::sync::Mutex<std::collections::HashMap>", reason = "Use DashMap instead" },
    { path = "std::sync::Arc<std::sync::Mutex<std::collections::HashMap>>", reason = "Use Arc<DashMap> instead" },
]

# 性能相关警告
too-many-arguments-threshold = 5
type-complexity-threshold = 100
```

**rustfmt 配置 (rustfmt.toml)：**
```toml
edition = "2024"
max_width = 100
hard_tabs = false
tab_spaces = 4
newline_style = "Unix"
use_small_heuristics = "Default"
reorder_imports = true
reorder_modules = true
remove_nested_parens = true
merge_derives = true
imports_granularity = "Crate"
group_imports = "StdExternalCrate"
```

### 2. 持续集成质量门禁

**GitHub Actions 配置：**
```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

env:
  CARGO_TERM_COLOR: always

jobs:
  test:
    name: Test
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - uses: actions/checkout@v4

    - uses: actions-rs/toolchain@v1
      with:
        profile: minimal
        toolchain: stable
        override: true
        components: clippy, rustfmt

    # 代码格式检查
    - name: Check formatting
      run: cargo fmt --all -- --check

    # Clippy 静态分析
    - name: Clippy
      run: cargo clippy --all-targets --all-features -- -D warnings

    # 单元测试
    - name: Run tests
      run: cargo test --all-features
      env:
        DATABASE_URL: postgres://postgres:postgres@localhost/postgres

    # 集成测试
    - name: Run integration tests
      run: cargo test --test integration_test
      env:
        DATABASE_URL: postgres://postgres:postgres@localhost/postgres

    # 代码覆盖率
    - name: Coverage
      run: |
        cargo install cargo-tarpaulin
        cargo tarpaulin --all-features --workspace --timeout 120 --out Xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3

    # 安全审计
    - name: Security audit
      run: |
        cargo install cargo-audit
        cargo audit

    # 基准测试
    - name: Benchmark
      run: cargo bench --no-run
```

### 3. 代码质量指标

**质量指标收集：**
```rust
// src/quality/metrics.rs
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct QualityMetrics {
    // 代码覆盖率
    pub test_coverage: f64,

    // 技术债务指标
    pub clippy_warnings: u32,
    pub complex_functions: u32,  // 圈复杂度 > 10
    pub long_functions: u32,     // 行数 > 150
    pub large_files: u32,        // 行数 > 500

    // 安全指标
    pub unsafe_blocks: u32,
    pub unwrap_usage: u32,
    pub security_vulnerabilities: u32,

    // 性能指标
    pub benchmark_regressions: u32,
    pub memory_leaks: u32,

    // 文档指标
    pub documentation_coverage: f64,
    pub missing_doc_warnings: u32,
}

impl QualityMetrics {
    pub fn calculate_quality_score(&self) -> f64 {
        let mut score = 100.0;

        // 测试覆盖率权重 30%
        score *= self.test_coverage / 100.0 * 0.3 + 0.7;

        // 技术债务惩罚
        score -= (self.clippy_warnings as f64 * 0.5).min(20.0);
        score -= (self.complex_functions as f64 * 2.0).min(20.0);

        // 安全问题严重惩罚
        score -= (self.unsafe_blocks as f64 * 5.0).min(30.0);
        score -= (self.unwrap_usage as f64 * 1.0).min(10.0);
        score -= (self.security_vulnerabilities as f64 * 10.0).min(50.0);

        score.max(0.0)
    }

    pub fn quality_gate_passed(&self) -> bool {
        self.calculate_quality_score() >= 80.0 &&
        self.test_coverage >= 80.0 &&
        self.security_vulnerabilities == 0 &&
        self.unwrap_usage == 0
    }
}

// 自动化质量报告
pub struct QualityReporter;

impl QualityReporter {
    pub fn generate_report(metrics: &QualityMetrics) -> String {
        format!(
            r#"
# Code Quality Report

## Overall Score: {:.1}/100

### Coverage
- Test Coverage: {:.1}%
- Documentation Coverage: {:.1}%

### Technical Debt
- Clippy Warnings: {}
- Complex Functions: {}
- Long Functions: {}
- Large Files: {}

### Security
- Unsafe Blocks: {}
- Unwrap Usage: {}
- Security Vulnerabilities: {}

### Quality Gate: {}
"#,
            metrics.calculate_quality_score(),
            metrics.test_coverage,
            metrics.documentation_coverage,
            metrics.clippy_warnings,
            metrics.complex_functions,
            metrics.long_functions,
            metrics.large_files,
            metrics.unsafe_blocks,
            metrics.unwrap_usage,
            metrics.security_vulnerabilities,
            if metrics.quality_gate_passed() { "✅ PASSED" } else { "❌ FAILED" }
        )
    }
}
```

## ✅ 生产就绪检查清单

### 1. 自动化检查清单

```rust
// src/quality/readiness.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct ProductionReadinessCheck {
    // 代码质量
    pub all_tests_pass: bool,
    pub coverage_above_80: bool,
    pub no_clippy_warnings: bool,
    pub no_unsafe_code: bool,

    // 安全性
    pub security_audit_clean: bool,
    pub input_validation_complete: bool,
    pub secrets_externalized: bool,
    pub tls_configured: bool,

    // 性能
    pub benchmarks_pass: bool,
    pub no_memory_leaks: bool,
    pub load_testing_complete: bool,

    // 运维
    pub logging_structured: bool,
    pub metrics_configured: bool,
    pub health_checks_implemented: bool,
    pub graceful_shutdown: bool,

    // 文档
    pub api_documented: bool,
    pub deployment_guide: bool,
    pub troubleshooting_guide: bool,
}

impl ProductionReadinessCheck {
    pub fn is_production_ready(&self) -> bool {
        self.all_tests_pass &&
        self.coverage_above_80 &&
        self.no_clippy_warnings &&
        self.security_audit_clean &&
        self.no_unsafe_code &&
        self.logging_structured &&
        self.health_checks_implemented
    }

    pub fn blocking_issues(&self) -> Vec<String> {
        let mut issues = Vec::new();

        if !self.all_tests_pass {
            issues.push("Tests are failing".to_string());
        }
        if !self.security_audit_clean {
            issues.push("Security vulnerabilities found".to_string());
        }
        if !self.no_unsafe_code {
            issues.push("Unsafe code blocks present".to_string());
        }

        issues
    }
}
```

这个质量保证体系确保 Claude Code 生成的代码都能达到生产级质量标准，通过自动化检查和持续监控维护高水准的代码质量。
