# ✅ 生产就绪检查清单

基于 Cursor Rust Rules 系统提炼的全面生产就绪验证清单，确保 Claude Code 协助开发的项目都能安全、稳定地部署到生产环境。

## 🚦 生产就绪分级标准

### 🟢 Level 1: 基础就绪 (必须)
- [ ] 所有测试通过
- [ ] 代码覆盖率 ≥ 80%
- [ ] 无 Clippy 警告
- [ ] 无安全漏洞
- [ ] 基础监控配置

### 🟡 Level 2: 运维就绪 (推荐)
- [ ] 结构化日志
- [ ] 健康检查端点
- [ ] 优雅关闭
- [ ] 性能基准通过
- [ ] 负载测试完成

### 🔴 Level 3: 企业就绪 (可选)
- [ ] 分布式追踪
- [ ] 混沌工程验证
- [ ] 多环境部署流水线
- [ ] 自动化回滚机制
- [ ] 全面监控和告警

## 📋 详细检查项目

### 1. 代码质量检查

#### 静态分析
```bash
# Clippy 检查 (必须零警告)
cargo clippy --all-targets --all-features -- -D warnings

# 格式检查
cargo fmt --all -- --check

# 安全审计
cargo audit

# 依赖检查
cargo deny check
```

**自动化验证脚本：**
```rust
// scripts/quality_check.rs
use std::process::Command;
use anyhow::{Result, Context};

pub struct QualityChecker;

impl QualityChecker {
    pub async fn run_all_checks() -> Result<QualityReport> {
        let mut report = QualityReport::new();

        report.clippy_passed = Self::run_clippy().await?;
        report.format_check_passed = Self::check_formatting().await?;
        report.security_audit_passed = Self::run_security_audit().await?;
        report.tests_passed = Self::run_tests().await?;
        report.coverage = Self::calculate_coverage().await?;

        Ok(report)
    }

    async fn run_clippy() -> Result<bool> {
        let output = Command::new("cargo")
            .args(&["clippy", "--all-targets", "--all-features", "--", "-D", "warnings"])
            .output()
            .context("Failed to run clippy")?;

        Ok(output.status.success())
    }

    async fn check_formatting() -> Result<bool> {
        let output = Command::new("cargo")
            .args(&["fmt", "--all", "--", "--check"])
            .output()
            .context("Failed to check formatting")?;

        Ok(output.status.success())
    }

    async fn run_security_audit() -> Result<bool> {
        let output = Command::new("cargo")
            .args(&["audit"])
            .output()
            .context("Failed to run security audit")?;

        Ok(output.status.success())
    }

    async fn run_tests() -> Result<bool> {
        let output = Command::new("cargo")
            .args(&["test", "--all-features"])
            .output()
            .context("Failed to run tests")?;

        Ok(output.status.success())
    }

    async fn calculate_coverage() -> Result<f64> {
        let output = Command::new("cargo")
            .args(&["tarpaulin", "--out", "Json"])
            .output()
            .context("Failed to calculate coverage")?;

        if output.status.success() {
            let coverage_data: serde_json::Value = serde_json::from_slice(&output.stdout)?;
            Ok(coverage_data["files"]["coverage"].as_f64().unwrap_or(0.0))
        } else {
            Ok(0.0)
        }
    }
}

#[derive(Debug)]
pub struct QualityReport {
    pub clippy_passed: bool,
    pub format_check_passed: bool,
    pub security_audit_passed: bool,
    pub tests_passed: bool,
    pub coverage: f64,
}

impl QualityReport {
    fn new() -> Self {
        Self {
            clippy_passed: false,
            format_check_passed: false,
            security_audit_passed: false,
            tests_passed: false,
            coverage: 0.0,
        }
    }

    pub fn is_production_ready(&self) -> bool {
        self.clippy_passed &&
        self.format_check_passed &&
        self.security_audit_passed &&
        self.tests_passed &&
        self.coverage >= 80.0
    }
}
```

#### 代码复杂度检查
```rust
// src/quality/complexity.rs
use syn::{File, Item, ItemFn, visit::Visit};
use std::path::Path;

pub struct ComplexityAnalyzer {
    max_function_lines: usize,
    max_cyclomatic_complexity: u32,
    max_file_lines: usize,
}

impl ComplexityAnalyzer {
    pub fn new() -> Self {
        Self {
            max_function_lines: 150,
            max_cyclomatic_complexity: 10,
            max_file_lines: 500,
        }
    }

    pub fn analyze_project(&self, src_dir: &Path) -> AnalysisReport {
        let mut report = AnalysisReport::new();

        // 遍历所有 Rust 文件
        for entry in walkdir::WalkDir::new(src_dir) {
            let entry = entry.unwrap();
            if entry.path().extension() == Some("rs".as_ref()) {
                if let Ok(analysis) = self.analyze_file(entry.path()) {
                    report.merge(analysis);
                }
            }
        }

        report
    }

    fn analyze_file(&self, file_path: &Path) -> anyhow::Result<FileAnalysis> {
        let content = std::fs::read_to_string(file_path)?;
        let syntax_tree = syn::parse_file(&content)?;

        let mut analysis = FileAnalysis::new(file_path.to_path_buf());
        analysis.total_lines = content.lines().count();

        // 检查文件大小
        if analysis.total_lines > self.max_file_lines {
            analysis.issues.push(format!("File too large: {} lines", analysis.total_lines));
        }

        // 分析函数
        for item in &syntax_tree.items {
            if let Item::Fn(func) = item {
                let func_analysis = self.analyze_function(func);
                analysis.functions.push(func_analysis);
            }
        }

        Ok(analysis)
    }

    fn analyze_function(&self, func: &ItemFn) -> FunctionAnalysis {
        let mut analysis = FunctionAnalysis {
            name: func.sig.ident.to_string(),
            line_count: self.count_function_lines(func),
            cyclomatic_complexity: self.calculate_complexity(func),
            issues: Vec::new(),
        };

        if analysis.line_count > self.max_function_lines {
            analysis.issues.push(format!("Function too long: {} lines", analysis.line_count));
        }

        if analysis.cyclomatic_complexity > self.max_cyclomatic_complexity {
            analysis.issues.push(format!("Function too complex: {} complexity", analysis.cyclomatic_complexity));
        }

        analysis
    }
}

#[derive(Debug)]
pub struct AnalysisReport {
    pub files: Vec<FileAnalysis>,
    pub total_issues: usize,
}

#[derive(Debug)]
pub struct FileAnalysis {
    pub path: std::path::PathBuf,
    pub total_lines: usize,
    pub functions: Vec<FunctionAnalysis>,
    pub issues: Vec<String>,
}

#[derive(Debug)]
pub struct FunctionAnalysis {
    pub name: String,
    pub line_count: usize,
    pub cyclomatic_complexity: u32,
    pub issues: Vec<String>,
}
```

### 2. 安全性检查

#### 输入验证检查
```rust
// src/security/validation.rs
use validator::{Validate, ValidationError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Validate)]
pub struct UserInput {
    #[validate(email, message = "Invalid email format")]
    pub email: String,

    #[validate(length(min = 8, max = 128, message = "Password must be 8-128 characters"))]
    #[validate(custom = "validate_password_strength")]
    pub password: String,

    #[validate(length(min = 1, max = 100))]
    pub name: String,

    #[validate(range(min = 0, max = 150))]
    pub age: Option<u32>,
}

fn validate_password_strength(password: &str) -> Result<(), ValidationError> {
    let has_uppercase = password.chars().any(|c| c.is_uppercase());
    let has_lowercase = password.chars().any(|c| c.is_lowercase());
    let has_digit = password.chars().any(|c| c.is_ascii_digit());
    let has_special = password.chars().any(|c| "!@#$%^&*()_+-=[]{}|;':\",./<>?".contains(c));

    if !(has_uppercase && has_lowercase && has_digit && has_special) {
        return Err(ValidationError::new("password_too_weak"));
    }

    // 检查常见弱密码
    let weak_patterns = ["123456", "password", "qwerty", "admin"];
    for pattern in &weak_patterns {
        if password.to_lowercase().contains(pattern) {
            return Err(ValidationError::new("password_too_common"));
        }
    }

    Ok(())
}

// 自动化安全检查
pub struct SecurityChecker;

impl SecurityChecker {
    pub fn check_input_validation(source_code: &str) -> Vec<SecurityIssue> {
        let mut issues = Vec::new();

        // 检查是否有未验证的用户输入
        if source_code.contains("request.into_inner()") &&
           !source_code.contains(".validate()") {
            issues.push(SecurityIssue::new(
                "Missing input validation",
                "User input should be validated before processing"
            ));
        }

        // 检查是否有 SQL 注入风险
        if source_code.contains("format!(\"SELECT") ||
           source_code.contains(&format!("INSERT")) {
            issues.push(SecurityIssue::new(
                "Potential SQL injection",
                "Use parameterized queries instead of string formatting"
            ));
        }

        issues
    }

    pub fn check_secrets_management(config_files: &[&str]) -> Vec<SecurityIssue> {
        let mut issues = Vec::new();

        for file_content in config_files {
            // 检查硬编码的密钥
            let secret_patterns = [
                r"password\s*=\s*['\"][^'\"]+['\"]",
                r"secret\s*=\s*['\"][^'\"]+['\"]",
                r"key\s*=\s*['\"][^'\"]+['\"]",
                r"token\s*=\s*['\"][^'\"]+['\"]",
            ];

            for pattern in &secret_patterns {
                if regex::Regex::new(pattern).unwrap().is_match(file_content) {
                    issues.push(SecurityIssue::new(
                        "Hardcoded secret detected",
                        "Secrets should be loaded from environment variables"
                    ));
                }
            }
        }

        issues
    }
}

#[derive(Debug)]
pub struct SecurityIssue {
    pub title: String,
    pub description: String,
    pub severity: SecuritySeverity,
}

#[derive(Debug)]
pub enum SecuritySeverity {
    Critical,
    High,
    Medium,
    Low,
}

impl SecurityIssue {
    pub fn new(title: &str, description: &str) -> Self {
        Self {
            title: title.to_string(),
            description: description.to_string(),
            severity: SecuritySeverity::High,
        }
    }
}
```

### 3. 性能和可扩展性检查

#### 性能基准测试
```rust
// benches/performance.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use my_app::{UserService, User};
use std::time::Duration;

fn benchmark_user_operations(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let service = rt.block_on(async {
        UserService::new_with_test_config().await
    });

    // 单个用户创建性能
    c.bench_function("create_user", |b| {
        b.to_async(&rt).iter(|| async {
            let user = User::new(
                black_box("test@example.com".to_string()),
                black_box("Test User".to_string()),
            );
            black_box(service.create_user(user).await)
        });
    });

    // 批量操作性能
    let mut group = c.benchmark_group("batch_operations");
    for size in [10, 100, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("create_users", size), size, |b, &size| {
            b.to_async(&rt).iter(|| async {
                let users: Vec<User> = (0..size)
                    .map(|i| User::new(format!("test{}@example.com", i), format!("User {}", i)))
                    .collect();
                black_box(service.create_users_batch(users).await)
            });
        });
    }
    group.finish();

    // 内存使用情况测试
    c.bench_function("memory_usage", |b| {
        b.iter(|| {
            let users: Vec<User> = (0..1000)
                .map(|i| User::new(format!("user{}@example.com", i), format!("User {}", i)))
                .collect();
            black_box(users)
        });
    });
}

// 性能回归检测
pub struct PerformanceMonitor;

impl PerformanceMonitor {
    pub fn check_performance_regression(
        current_results: &BenchmarkResults,
        baseline: &BenchmarkResults,
    ) -> PerformanceReport {
        let mut report = PerformanceReport::new();

        for (test_name, current) in &current_results.tests {
            if let Some(baseline_result) = baseline.tests.get(test_name) {
                let regression_ratio = current.duration.as_secs_f64() / baseline_result.duration.as_secs_f64();

                if regression_ratio > 1.2 {  // 20% 性能下降
                    report.regressions.push(PerformanceRegression {
                        test_name: test_name.clone(),
                        current_duration: current.duration,
                        baseline_duration: baseline_result.duration,
                        regression_ratio,
                    });
                }
            }
        }

        report
    }
}

#[derive(Debug)]
pub struct BenchmarkResults {
    pub tests: std::collections::HashMap<String, BenchmarkResult>,
}

#[derive(Debug)]
pub struct BenchmarkResult {
    pub duration: Duration,
    pub memory_usage: usize,
}

#[derive(Debug)]
pub struct PerformanceReport {
    pub regressions: Vec<PerformanceRegression>,
}

#[derive(Debug)]
pub struct PerformanceRegression {
    pub test_name: String,
    pub current_duration: Duration,
    pub baseline_duration: Duration,
    pub regression_ratio: f64,
}

criterion_group!(benches, benchmark_user_operations);
criterion_main!(benches);
```

### 4. 运维就绪检查

#### 健康检查和监控
```rust
// src/health/mod.rs
use axum::{response::Json, http::StatusCode};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::time::Instant;

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub timestamp: String,
    pub version: String,
    pub checks: Vec<HealthCheck>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthCheck {
    pub name: String,
    pub status: String,
    pub response_time_ms: u64,
    pub details: Option<serde_json::Value>,
}

pub struct HealthChecker {
    db_pool: PgPool,
}

impl HealthChecker {
    pub fn new(db_pool: PgPool) -> Self {
        Self { db_pool }
    }

    pub async fn check_health(&self) -> (StatusCode, Json<HealthStatus>) {
        let start = Instant::now();
        let mut checks = Vec::new();
        let mut overall_healthy = true;

        // 数据库健康检查
        let db_check = self.check_database().await;
        overall_healthy &= db_check.status == "healthy";
        checks.push(db_check);

        // Redis 健康检查
        let redis_check = self.check_redis().await;
        overall_healthy &= redis_check.status == "healthy";
        checks.push(redis_check);

        // 依赖服务检查
        let external_check = self.check_external_services().await;
        overall_healthy &= external_check.status == "healthy";
        checks.push(external_check);

        let status = HealthStatus {
            status: if overall_healthy { "healthy".to_string() } else { "unhealthy".to_string() },
            timestamp: chrono::Utc::now().to_rfc3339(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            checks,
        };

        let status_code = if overall_healthy {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        };

        (status_code, Json(status))
    }

    async fn check_database(&self) -> HealthCheck {
        let start = Instant::now();

        match sqlx::query("SELECT 1").fetch_one(&self.db_pool).await {
            Ok(_) => HealthCheck {
                name: "database".to_string(),
                status: "healthy".to_string(),
                response_time_ms: start.elapsed().as_millis() as u64,
                details: None,
            },
            Err(e) => HealthCheck {
                name: "database".to_string(),
                status: "unhealthy".to_string(),
                response_time_ms: start.elapsed().as_millis() as u64,
                details: Some(serde_json::json!({"error": e.to_string()})),
            },
        }
    }

    async fn check_redis(&self) -> HealthCheck {
        let start = Instant::now();

        // Redis 连接检查逻辑
        HealthCheck {
            name: "redis".to_string(),
            status: "healthy".to_string(),
            response_time_ms: start.elapsed().as_millis() as u64,
            details: None,
        }
    }

    async fn check_external_services(&self) -> HealthCheck {
        let start = Instant::now();

        // 外部服务检查逻辑
        HealthCheck {
            name: "external_services".to_string(),
            status: "healthy".to_string(),
            response_time_ms: start.elapsed().as_millis() as u64,
            details: None,
        }
    }
}

// 优雅关闭实现
pub struct GracefulShutdown {
    shutdown_timeout: std::time::Duration,
}

impl GracefulShutdown {
    pub fn new(timeout_seconds: u64) -> Self {
        Self {
            shutdown_timeout: std::time::Duration::from_secs(timeout_seconds),
        }
    }

    pub async fn wait_for_shutdown_signal(&self) {
        let ctrl_c = async {
            tokio::signal::ctrl_c()
                .await
                .expect("failed to install Ctrl+C handler");
        };

        #[cfg(unix)]
        let terminate = async {
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("failed to install signal handler")
                .recv()
                .await;
        };

        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();

        tokio::select! {
            _ = ctrl_c => {},
            _ = terminate => {},
        }

        tracing::info!("Shutdown signal received, starting graceful shutdown");
    }

    pub async fn shutdown_gracefully<F, Fut>(&self, cleanup_fn: F)
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = ()>,
    {
        // 设置关闭超时
        tokio::select! {
            _ = cleanup_fn() => {
                tracing::info!("Graceful shutdown completed");
            }
            _ = tokio::time::sleep(self.shutdown_timeout) => {
                tracing::warn!("Graceful shutdown timed out, forcing exit");
            }
        }
    }
}
```

### 5. 部署和 CI/CD 检查

#### Docker 容器化检查
```dockerfile
# Dockerfile - 多阶段构建优化
FROM rust:1.75 as builder

WORKDIR /app
COPY . .

# 缓存依赖构建
RUN cargo build --release

# 运行时镜像
FROM debian:bookworm-slim

# 安装运行时依赖
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 创建非 root 用户
RUN useradd -m -u 1001 appuser

WORKDIR /app
COPY --from=builder /app/target/release/my-app /app/
COPY --from=builder /app/config /app/config

# 设置权限
RUN chown -R appuser:appuser /app
USER appuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080

CMD ["./my-app"]
```

#### Kubernetes 部署配置
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
  labels:
    app: my-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: my-app
        image: my-app:latest
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: database-url
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: jwt-secret

        # 资源限制
        resources:
          limits:
            memory: "512Mi"
            cpu: "500m"
          requests:
            memory: "256Mi"
            cpu: "250m"

        # 健康检查
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10

        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5

        # 安全上下文
        securityContext:
          runAsNonRoot: true
          runAsUser: 1001
          readOnlyRootFilesystem: true
          allowPrivilegeEscalation: false
          capabilities:
            drop:
            - ALL
```

## 🎯 自动化生产就绪验证

```rust
// scripts/production_readiness.rs
use anyhow::Result;
use std::collections::HashMap;

pub struct ProductionReadinessValidator;

impl ProductionReadinessValidator {
    pub async fn validate_full_readiness() -> Result<ReadinessReport> {
        let mut report = ReadinessReport::new();

        // Level 1: 基础检查
        report.basic_checks = Self::run_basic_checks().await?;

        // Level 2: 运维检查
        report.operational_checks = Self::run_operational_checks().await?;

        // Level 3: 企业级检查
        report.enterprise_checks = Self::run_enterprise_checks().await?;

        // 计算总体就绪度
        report.overall_readiness = Self::calculate_readiness_score(&report);

        Ok(report)
    }

    async fn run_basic_checks() -> Result<HashMap<String, bool>> {
        let mut checks = HashMap::new();

        checks.insert("tests_pass".to_string(), Self::check_tests().await?);
        checks.insert("coverage_80".to_string(), Self::check_coverage().await?);
        checks.insert("no_clippy_warnings".to_string(), Self::check_clippy().await?);
        checks.insert("security_audit".to_string(), Self::check_security().await?);
        checks.insert("no_unsafe_code".to_string(), Self::check_unsafe_code().await?);

        Ok(checks)
    }

    async fn run_operational_checks() -> Result<HashMap<String, bool>> {
        let mut checks = HashMap::new();

        checks.insert("structured_logging".to_string(), Self::check_logging().await?);
        checks.insert("health_checks".to_string(), Self::check_health_endpoints().await?);
        checks.insert("graceful_shutdown".to_string(), Self::check_graceful_shutdown().await?);
        checks.insert("performance_benchmarks".to_string(), Self::check_benchmarks().await?);
        checks.insert("load_testing".to_string(), Self::check_load_testing().await?);

        Ok(checks)
    }

    fn calculate_readiness_score(report: &ReadinessReport) -> ReadinessLevel {
        let basic_passed = report.basic_checks.values().all(|&v| v);
        let operational_passed = report.operational_checks.values().filter(|&&v| v).count() >= 3;
        let enterprise_passed = report.enterprise_checks.values().filter(|&&v| v).count() >= 2;

        if basic_passed && operational_passed && enterprise_passed {
            ReadinessLevel::Enterprise
        } else if basic_passed && operational_passed {
            ReadinessLevel::Operational
        } else if basic_passed {
            ReadinessLevel::Basic
        } else {
            ReadinessLevel::NotReady
        }
    }
}

#[derive(Debug)]
pub struct ReadinessReport {
    pub basic_checks: HashMap<String, bool>,
    pub operational_checks: HashMap<String, bool>,
    pub enterprise_checks: HashMap<String, bool>,
    pub overall_readiness: ReadinessLevel,
}

#[derive(Debug)]
pub enum ReadinessLevel {
    NotReady,
    Basic,
    Operational,
    Enterprise,
}

impl ReadinessReport {
    pub fn new() -> Self {
        Self {
            basic_checks: HashMap::new(),
            operational_checks: HashMap::new(),
            enterprise_checks: HashMap::new(),
            overall_readiness: ReadinessLevel::NotReady,
        }
    }

    pub fn print_report(&self) {
        println!("🚀 Production Readiness Report\n");
        println!("Overall Level: {:?}\n", self.overall_readiness);

        println!("🟢 Basic Checks (Required):");
        for (check, passed) in &self.basic_checks {
            let icon = if *passed { "✅" } else { "❌" };
            println!("  {} {}", icon, check);
        }

        println!("\n🟡 Operational Checks (Recommended):");
        for (check, passed) in &self.operational_checks {
            let icon = if *passed { "✅" } else { "❌" };
            println!("  {} {}", icon, check);
        }

        println!("\n🔴 Enterprise Checks (Optional):");
        for (check, passed) in &self.enterprise_checks {
            let icon = if *passed { "✅" } else { "❌" };
            println!("  {} {}", icon, check);
        }
    }
}
```

## 📊 持续监控和告警

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'my-app'
    static_configs:
      - targets: ['my-app:8080']
    metrics_path: /metrics
    scrape_interval: 10s

# 告警规则
rule_files:
  - "alert_rules.yml"

---
# alert_rules.yml
groups:
- name: my-app
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} errors per second"

  - alert: HighMemoryUsage
    expr: process_resident_memory_bytes / 1024 / 1024 > 500
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage"
      description: "Memory usage is {{ $value }}MB"
```

这个生产就绪检查清单确保 Claude Code 协助开发的项目都能安全、稳定、高效地在生产环境中运行。
