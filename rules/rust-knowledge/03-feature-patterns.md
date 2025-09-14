# 🔧 功能模式库

基于 Cursor Rust Rules 功能层规则整理的常用功能实现模式，为 Claude Code 提供可复用的代码模式和最佳实践参考。

## 🌐 Web 服务模式 (Axum)

### 1. 现代 Axum 0.8+ 服务架构

**完整的 Web 服务脚手架：**
```rust
// src/lib.rs - 核心应用结构
use axum::{
    extract::{Query, Path, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use sqlx::PgPool;
use std::sync::Arc;
use utoipa::{OpenApi, ToSchema};
use utoipa_swagger_ui::SwaggerUi;

#[derive(Clone)]
pub struct AppState {
    pub db: PgPool,
    pub config: Arc<AppConfig>,
}

#[derive(OpenApi)]
#[openapi(
    paths(
        handlers::users::list_users,
        handlers::users::create_user,
        handlers::users::get_user,
    ),
    components(schemas(User, CreateUserRequest, ListUsersQuery))
)]
struct ApiDoc;

pub fn create_app(state: AppState) -> Router {
    let api_routes = Router::new()
        .route("/users", get(handlers::users::list_users).post(handlers::users::create_user))
        .route("/users/:id", get(handlers::users::get_user))
        .with_state(state.clone());

    Router::new()
        .route("/health", get(health_check))
        .nest("/api/v1", api_routes)
        .merge(SwaggerUi::new("/swagger-ui").url("/api-doc/openapi.json", ApiDoc::openapi()))
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(CorsLayer::permissive())
                .layer(CompressionLayer::new())
        )
}

async fn health_check() -> &'static str {
    "OK"
}
```

**分层处理器模式：**
```rust
// src/handlers/users.rs - 请求处理器
use axum::{extract::{Query, Path, State}, response::Json, http::StatusCode};
use utoipa::ToSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

#[derive(Debug, Deserialize, ToSchema)]
pub struct ListUsersQuery {
    #[serde(default = "default_page")]
    pub page: u32,
    #[serde(default = "default_limit")]
    pub limit: u32,
}

fn default_page() -> u32 { 1 }
fn default_limit() -> u32 { 20 }

#[derive(Debug, Deserialize, Validate, ToSchema)]
pub struct CreateUserRequest {
    #[validate(email)]
    pub email: String,
    #[validate(length(min = 2, max = 50))]
    pub name: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct User {
    pub id: uuid::Uuid,
    pub email: String,
    pub name: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// List all users with pagination
#[utoipa::path(
    get,
    path = "/api/v1/users",
    params(ListUsersQuery),
    responses(
        (status = 200, description = "List of users", body = Vec<User>)
    )
)]
pub async fn list_users(
    Query(params): Query<ListUsersQuery>,
    State(state): State<AppState>,
) -> Result<Json<Vec<User>>, StatusCode> {
    let offset = (params.page.saturating_sub(1)) * params.limit;

    match sqlx::query_as!(
        User,
        "SELECT id, email, name, created_at FROM users ORDER BY created_at DESC LIMIT $1 OFFSET $2",
        params.limit as i64,
        offset as i64
    )
    .fetch_all(&state.db)
    .await
    {
        Ok(users) => Ok(Json(users)),
        Err(e) => {
            tracing::error!("Database error: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Create a new user
#[utoipa::path(
    post,
    path = "/api/v1/users",
    request_body = CreateUserRequest,
    responses(
        (status = 201, description = "User created", body = User),
        (status = 400, description = "Bad request")
    )
)]
pub async fn create_user(
    State(state): State<AppState>,
    Json(request): Json<CreateUserRequest>,
) -> Result<(StatusCode, Json<User>), StatusCode> {
    if let Err(_) = request.validate() {
        return Err(StatusCode::BAD_REQUEST);
    }

    match sqlx::query_as!(
        User,
        "INSERT INTO users (id, email, name) VALUES ($1, $2, $3) RETURNING *",
        uuid::Uuid::new_v4(),
        request.email,
        request.name
    )
    .fetch_one(&state.db)
    .await
    {
        Ok(user) => Ok((StatusCode::CREATED, Json(user))),
        Err(e) => {
            tracing::error!("Failed to create user: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}
```

### 2. 中间件和安全模式

**JWT 认证中间件：**
```rust
// src/middleware/auth.rs
use axum::{
    extract::{Request, State},
    http::{header::AUTHORIZATION, StatusCode},
    middleware::Next,
    response::Response,
};
use jsonwebtoken::{decode, DecodingKey, Validation, Algorithm};

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct Claims {
    pub user_id: uuid::Uuid,
    pub email: String,
    pub exp: usize,
}

pub async fn auth_middleware(
    State(state): State<AppState>,
    mut request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let auth_header = request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|header| header.to_str().ok())
        .ok_or(StatusCode::UNAUTHORIZED)?;

    let token = auth_header
        .strip_prefix("Bearer ")
        .ok_or(StatusCode::UNAUTHORIZED)?;

    let claims = decode::<Claims>(
        token,
        &DecodingKey::from_secret(state.config.jwt_secret.as_ref()),
        &Validation::new(Algorithm::HS256),
    )
    .map_err(|_| StatusCode::UNAUTHORIZED)?
    .claims;

    // 将用户信息注入请求扩展
    request.extensions_mut().insert(claims);

    Ok(next.run(request).await)
}

// 提取认证用户的便捷提取器
#[axum::async_trait]
impl<S> axum::extract::FromRequestParts<S> for Claims
where
    S: Send + Sync,
{
    type Rejection = StatusCode;

    async fn from_request_parts(
        parts: &mut axum::http::request::Parts,
        _state: &S,
    ) -> Result<Self, Self::Rejection> {
        parts
            .extensions
            .get::<Claims>()
            .cloned()
            .ok_or(StatusCode::UNAUTHORIZED)
    }
}
```

## 🖥️ CLI 应用模式 (Clap)

### 1. 现代 Clap 4.0+ 命令结构

**多层级子命令模式：**
```rust
// src/cli.rs - 命令定义
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "my-tool")]
#[command(about = "A powerful CLI tool for development")]
#[command(version)]
pub struct Cli {
    #[arg(short, long, global = true)]
    pub verbose: bool,

    #[arg(short, long, global = true, value_name = "FILE")]
    pub config: Option<PathBuf>,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Initialize a new project
    Init {
        #[arg(value_name = "DIR")]
        path: PathBuf,

        #[arg(short, long, value_enum, default_value = "web")]
        template: ProjectTemplate,

        #[arg(long)]
        no_git: bool,
    },

    /// Build the project
    Build {
        #[arg(short, long)]
        release: bool,

        #[arg(short, long)]
        target: Option<String>,
    },

    /// Deploy the application
    Deploy {
        #[arg(short, long, value_enum, default_value = "staging")]
        environment: Environment,

        #[arg(long)]
        dry_run: bool,
    },

    /// Manage configurations
    Config {
        #[command(subcommand)]
        command: ConfigCommands,
    },
}

#[derive(Subcommand)]
pub enum ConfigCommands {
    /// Show current configuration
    Show,
    /// Set a configuration value
    Set {
        key: String,
        value: String,
    },
    /// Remove a configuration value
    Remove {
        key: String,
    },
}

#[derive(ValueEnum, Clone, Debug)]
pub enum ProjectTemplate {
    Web,
    Cli,
    Library,
    Grpc,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum Environment {
    Development,
    Staging,
    Production,
}
```

**命令处理器模式：**
```rust
// src/commands/mod.rs - 命令处理
use crate::cli::{Commands, ConfigCommands, ProjectTemplate, Environment};
use anyhow::Result;

pub mod init;
pub mod build;
pub mod deploy;
pub mod config;

pub async fn handle_command(command: Commands) -> Result<()> {
    match command {
        Commands::Init { path, template, no_git } => {
            init::handle_init(path, template, no_git).await
        }
        Commands::Build { release, target } => {
            build::handle_build(release, target).await
        }
        Commands::Deploy { environment, dry_run } => {
            deploy::handle_deploy(environment, dry_run).await
        }
        Commands::Config { command } => {
            handle_config_command(command).await
        }
    }
}

async fn handle_config_command(command: ConfigCommands) -> Result<()> {
    match command {
        ConfigCommands::Show => config::show().await,
        ConfigCommands::Set { key, value } => config::set(&key, &value).await,
        ConfigCommands::Remove { key } => config::remove(&key).await,
    }
}
```

**进度条和用户交互：**
```rust
// src/commands/build.rs - 构建命令实现
use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use tokio::process::Command;
use std::time::Duration;

pub async fn handle_build(release: bool, target: Option<String>) -> Result<()> {
    println!("🔨 Building project...");

    // 创建进度条
    let pb = ProgressBar::new_spinner();
    pb.set_style(ProgressStyle::default_spinner()
        .template("{spinner:.green} [{elapsed_precise}] {msg}")
        .unwrap());
    pb.set_message("Compiling...");

    // 启动进度条
    let pb_clone = pb.clone();
    let progress_task = tokio::spawn(async move {
        loop {
            pb_clone.tick();
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    });

    // 构建命令
    let mut cmd = Command::new("cargo");
    cmd.arg("build");

    if release {
        cmd.arg("--release");
    }

    if let Some(target) = target {
        cmd.args(&["--target", &target]);
    }

    // 执行构建
    let output = cmd.output().await?;

    // 停止进度条
    progress_task.abort();
    pb.finish_with_message("Build completed!");

    if output.status.success() {
        println!("✅ Build successful!");
        Ok(())
    } else {
        let error = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Build failed: {}", error);
    }
}
```

## 🚀 gRPC 服务模式 (Tonic)

### 1. Protocol Buffer 和服务定义

**proto 文件组织：**
```proto
// proto/user.proto
syntax = "proto3";
package user.v1;

service UserService {
  rpc GetUser(GetUserRequest) returns (GetUserResponse);
  rpc ListUsers(ListUsersRequest) returns (ListUsersResponse);
  rpc CreateUser(CreateUserRequest) returns (CreateUserResponse);
  rpc UpdateUser(UpdateUserRequest) returns (UpdateUserResponse);
  rpc DeleteUser(DeleteUserRequest) returns (DeleteUserResponse);
}

message User {
  string id = 1;
  string email = 2;
  string name = 3;
  string created_at = 4;
  string updated_at = 5;
}

message GetUserRequest {
  string id = 1;
}

message GetUserResponse {
  User user = 1;
}

message ListUsersRequest {
  int32 page = 1;
  int32 limit = 2;
}

message ListUsersResponse {
  repeated User users = 1;
  int32 total = 2;
}
```

**Inner Types 模式实现：**
```rust
// src/models/user.rs - 内部数据模型
use chrono::{DateTime, Utc};
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, sqlx::FromRow)]
pub struct User {
    pub id: Uuid,
    pub email: String,
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

// MessageSanitizer trait 确保数据清理
pub trait MessageSanitizer {
    fn sanitize(self) -> Self;
}

impl MessageSanitizer for User {
    fn sanitize(mut self) -> Self {
        self.email = self.email.trim().to_lowercase();
        self.name = self.name.trim().to_string();
        self
    }
}

// Protocol Buffer 转换实现
impl From<User> for proto::user::v1::User {
    fn from(user: User) -> Self {
        Self {
            id: user.id.to_string(),
            email: user.email,
            name: user.name,
            created_at: user.created_at.to_rfc3339(),
            updated_at: user.updated_at.to_rfc3339(),
        }
    }
}

impl TryFrom<proto::user::v1::CreateUserRequest> for User {
    type Error = anyhow::Error;

    fn try_from(request: proto::user::v1::CreateUserRequest) -> Result<Self, Self::Error> {
        Ok(User {
            id: Uuid::new_v4(),
            email: request.email,
            name: request.name,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }.sanitize())
    }
}
```

**gRPC 服务实现：**
```rust
// src/grpc/user_service.rs
use tonic::{Request, Response, Status};
use tracing::{info, error};
use std::sync::Arc;

pub struct UserServiceImpl {
    repository: Arc<UserRepository>,
}

impl UserServiceImpl {
    pub fn new(repository: Arc<UserRepository>) -> Self {
        Self { repository }
    }
}

#[tonic::async_trait]
impl proto::user::v1::user_service_server::UserService for UserServiceImpl {
    async fn get_user(
        &self,
        request: Request<proto::user::v1::GetUserRequest>,
    ) -> Result<Response<proto::user::v1::GetUserResponse>, Status> {
        let req = request.into_inner();
        info!("Getting user with id: {}", req.id);

        let user_id = uuid::Uuid::parse_str(&req.id)
            .map_err(|_| Status::invalid_argument("Invalid user ID format"))?;

        match self.repository.find_by_id(user_id).await {
            Ok(Some(user)) => {
                let response = proto::user::v1::GetUserResponse {
                    user: Some(user.into()),
                };
                Ok(Response::new(response))
            }
            Ok(None) => Err(Status::not_found("User not found")),
            Err(e) => {
                error!("Database error: {}", e);
                Err(Status::internal("Internal server error"))
            }
        }
    }

    async fn create_user(
        &self,
        request: Request<proto::user::v1::CreateUserRequest>,
    ) -> Result<Response<proto::user::v1::CreateUserResponse>, Status> {
        let req = request.into_inner();
        info!("Creating user: {}", req.email);

        let user = User::try_from(req)
            .map_err(|e| Status::invalid_argument(format!("Invalid user data: {}", e)))?;

        match self.repository.create(user).await {
            Ok(created_user) => {
                let response = proto::user::v1::CreateUserResponse {
                    user: Some(created_user.into()),
                };
                Ok(Response::new(response))
            }
            Err(e) => {
                error!("Failed to create user: {}", e);
                Err(Status::internal("Failed to create user"))
            }
        }
    }
}
```

## 🗄️ 数据库模式 (SQLx)

### 1. 仓储模式实现

**异步仓储 trait：**
```rust
// src/repositories/traits.rs
use async_trait::async_trait;

#[async_trait]
pub trait Repository<T, K> {
    type Error: std::error::Error + Send + Sync + 'static;

    async fn create(&self, entity: T) -> Result<T, Self::Error>;
    async fn find_by_id(&self, id: K) -> Result<Option<T>, Self::Error>;
    async fn find_all(&self) -> Result<Vec<T>, Self::Error>;
    async fn update(&self, entity: T) -> Result<T, Self::Error>;
    async fn delete(&self, id: K) -> Result<(), Self::Error>;
}

// 用户仓储具体实现
use sqlx::{PgPool, postgres::PgRow, Row};

pub struct UserRepository {
    pool: PgPool,
}

impl UserRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl Repository<User, Uuid> for UserRepository {
    type Error = sqlx::Error;

    async fn create(&self, user: User) -> Result<User, Self::Error> {
        sqlx::query_as!(
            User,
            r#"
            INSERT INTO users (id, email, name, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING *
            "#,
            user.id,
            user.email,
            user.name,
            user.created_at,
            user.updated_at
        )
        .fetch_one(&self.pool)
        .await
    }

    async fn find_by_id(&self, id: Uuid) -> Result<Option<User>, Self::Error> {
        sqlx::query_as!(
            User,
            "SELECT * FROM users WHERE id = $1",
            id
        )
        .fetch_optional(&self.pool)
        .await
    }
}
```

### 2. 事务管理模式

**事务协调器：**
```rust
// src/services/user_service.rs - 事务管理
use sqlx::{PgPool, Postgres, Transaction};

pub struct UserService {
    pool: PgPool,
}

impl UserService {
    pub async fn create_user_with_profile(
        &self,
        user_data: CreateUserRequest,
        profile_data: CreateProfileRequest,
    ) -> anyhow::Result<(User, UserProfile)> {
        let mut tx = self.pool.begin().await?;

        // 在事务中创建用户
        let user = self.create_user_tx(&mut tx, user_data).await?;

        // 在同一事务中创建用户档案
        let profile = self.create_profile_tx(&mut tx, profile_data, user.id).await?;

        // 提交事务
        tx.commit().await?;

        Ok((user, profile))
    }

    async fn create_user_tx(
        &self,
        tx: &mut Transaction<'_, Postgres>,
        data: CreateUserRequest,
    ) -> anyhow::Result<User> {
        sqlx::query_as!(
            User,
            "INSERT INTO users (id, email, name) VALUES ($1, $2, $3) RETURNING *",
            uuid::Uuid::new_v4(),
            data.email,
            data.name
        )
        .fetch_one(&mut **tx)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to create user: {}", e))
    }
}
```

## ⚙️ 配置管理模式 (Figment)

### 1. 多源配置合并

**灵活的配置系统：**
```rust
// src/config.rs
use figment::{Figment, providers::{Format, Yaml, Toml, Json, Env}};
use serde::{Deserialize, Serialize};
use validator::Validate;

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct AppConfig {
    #[validate(range(min = 1, max = 65535))]
    pub port: u16,

    #[validate(url)]
    pub database_url: String,

    #[validate(length(min = 32))]
    pub jwt_secret: String,

    #[serde(default)]
    pub debug: bool,

    #[validate(nested)]
    pub redis: RedisConfig,

    #[validate(nested)]
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct RedisConfig {
    #[validate(url)]
    pub url: String,

    #[validate(range(min = 1, max = 1000))]
    pub pool_size: u32,
}

impl AppConfig {
    pub fn load() -> anyhow::Result<Self> {
        let config = Figment::new()
            // 默认配置文件
            .merge(Yaml::file("config/default.yaml"))
            .merge(Toml::file("config/default.toml"))

            // 环境特定配置
            .merge(Yaml::file(format!("config/{}.yaml", Self::environment())))
            .merge(Toml::file(format!("config/{}.toml", Self::environment())))

            // 环境变量覆盖 (最高优先级)
            .merge(Env::prefixed("APP_").split("_"))

            .extract()?;

        // 验证配置
        config.validate()
            .map_err(|e| anyhow::anyhow!("Configuration validation failed: {}", e))?;

        Ok(config)
    }

    fn environment() -> String {
        std::env::var("APP_ENV")
            .or_else(|_| std::env::var("ENVIRONMENT"))
            .unwrap_or_else(|_| "development".to_string())
    }
}
```

### 2. 配置热重载

**实时配置更新：**
```rust
// src/config/manager.rs
use arc_swap::ArcSwap;
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use std::sync::Arc;
use tokio::sync::broadcast;

pub struct ConfigManager<T> {
    current: Arc<ArcSwap<T>>,
    reload_tx: broadcast::Sender<Arc<T>>,
}

impl<T> ConfigManager<T>
where
    T: for<'de> serde::Deserialize<'de> + validator::Validate + Clone + Send + Sync + 'static,
{
    pub fn new(initial_config: T) -> Self {
        let (reload_tx, _) = broadcast::channel(16);
        let current = Arc::new(ArcSwap::from_pointee(initial_config));

        Self {
            current,
            reload_tx,
        }
    }

    pub fn get(&self) -> arc_swap::Guard<Arc<T>> {
        self.current.load()
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Arc<T>> {
        self.reload_tx.subscribe()
    }

    pub async fn start_watching(&self, config_path: &std::path::Path) -> anyhow::Result<()> {
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);

        let mut watcher = RecommendedWatcher::new(
            move |res| {
                if let Ok(event) = res {
                    if event.kind.is_modify() {
                        let _ = tx.try_send(());
                    }
                }
            },
            notify::Config::default(),
        )?;

        watcher.watch(config_path, RecursiveMode::NonRecursive)?;

        let current = self.current.clone();
        let reload_tx = self.reload_tx.clone();
        let config_path = config_path.to_owned();

        tokio::spawn(async move {
            while rx.recv().await.is_some() {
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

                match AppConfig::load() {
                    Ok(new_config) => {
                        let new_config = Arc::new(new_config);
                        current.store(new_config.clone());

                        if let Err(e) = reload_tx.send(new_config) {
                            tracing::warn!("Failed to notify config subscribers: {}", e);
                        } else {
                            tracing::info!("Configuration reloaded successfully");
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to reload configuration: {}", e);
                    }
                }
            }
        });

        // 保持 watcher 存活
        std::mem::forget(watcher);
        Ok(())
    }
}
```

这些功能模式为 Claude Code 提供了丰富的、经过生产验证的代码模板，确保能够快速构建高质量的 Rust 应用程序。
