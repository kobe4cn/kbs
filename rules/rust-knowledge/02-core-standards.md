# ⚡ 核心开发标准

基于 Cursor Rust Rules 核心规则提炼的 Rust 2024 开发标准，确保所有项目都遵循一致的高质量代码实践。

## 🏗️ 代码质量核心原则

### 1. Rust 2024 版本标准

**强制要求：**
```toml
[package]
name = "my_project"
version = "0.1.0"
edition = "2024"                  # 必须使用 Rust 2024

[dependencies]
# 使用最新稳定版本的 crate
tokio = "1.0"                     # 不使用过时版本
serde = { version = "1.0", features = ["derive"] }
```

**现代语言特性应用：**
```rust
// ✅ 使用现代 Rust 特性
#![deny(clippy::all)]
#![warn(clippy::pedantic)]

// 异步函数优先
pub async fn process_data(input: Vec<String>) -> anyhow::Result<ProcessedData> {
    // 使用 ? 操作符进行错误处理
    let validated = validate_input(&input)?;
    let processed = transform_data(validated).await?;
    Ok(processed)
}

// 使用强类型和 newtype 模式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UserId(uuid::Uuid);

#[derive(Debug, Clone, PartialEq)]
pub struct Email(String);

impl Email {
    pub fn new(email: String) -> anyhow::Result<Self> {
        if email.contains('@') && email.len() > 5 {
            Ok(Email(email))
        } else {
            Err(anyhow::anyhow!("Invalid email format"))
        }
    }
}
```

### 2. DRY 和 SRP 原则实施

**不重复原则 (DRY)：**
```rust
// ❌ 重复的验证逻辑
fn validate_user_email(email: &str) -> bool {
    email.contains('@') && email.len() > 5
}

fn validate_admin_email(email: &str) -> bool {
    email.contains('@') && email.len() > 5
}

// ✅ 统一的验证逻辑
trait EmailValidator {
    fn validate_email(&self, email: &str) -> anyhow::Result<()> {
        if email.contains('@') && email.len() > 5 {
            Ok(())
        } else {
            Err(anyhow::anyhow!("Invalid email format"))
        }
    }
}

struct UserValidator;
struct AdminValidator;

impl EmailValidator for UserValidator {}
impl EmailValidator for AdminValidator {}
```

**单一职责原则 (SRP)：**
```rust
// ✅ 每个结构体只有单一职责
pub struct UserRepository {
    pool: sqlx::PgPool,
}

impl UserRepository {
    pub async fn create_user(&self, user: NewUser) -> sqlx::Result<User> {
        // 只负责数据库操作
    }

    pub async fn find_by_email(&self, email: &str) -> sqlx::Result<Option<User>> {
        // 只负责查询操作
    }
}

pub struct UserService {
    repository: Arc<UserRepository>,
    email_service: Arc<EmailService>,
}

impl UserService {
    pub async fn register_user(&self, request: RegisterRequest) -> anyhow::Result<User> {
        // 只负责业务逻辑协调
        let user = self.repository.create_user(request.into()).await?;
        self.email_service.send_welcome_email(&user).await?;
        Ok(user)
    }
}
```

### 3. 功能导向的文件组织

**按功能而非类型组织：**
```
// ❌ 按类型组织 (不推荐)
src/
├── models/
│   ├── user.rs
│   ├── product.rs
│   └── order.rs
├── controllers/
│   ├── user_controller.rs
│   ├── product_controller.rs
│   └── order_controller.rs
└── services/
    ├── user_service.rs
    ├── product_service.rs
    └── order_service.rs

// ✅ 按功能组织 (推荐)
src/
├── users/
│   ├── mod.rs
│   ├── model.rs
│   ├── service.rs
│   ├── repository.rs
│   └── handlers.rs
├── products/
│   ├── mod.rs
│   ├── model.rs
│   ├── service.rs
│   ├── repository.rs
│   └── handlers.rs
└── orders/
    ├── mod.rs
    ├── model.rs
    ├── service.rs
    ├── repository.rs
    └── handlers.rs
```

### 4. 文件大小和函数长度控制

**严格的大小限制：**
```rust
// ✅ 函数长度控制 (≤ 150 行)
pub async fn process_order(
    order_request: OrderRequest,
    user_service: &UserService,
    product_service: &ProductService,
    payment_service: &PaymentService,
) -> anyhow::Result<Order> {
    // Step 1: 验证用户
    let user = user_service.validate_user(&order_request.user_id).await?;

    // Step 2: 验证产品
    let products = product_service.validate_products(&order_request.items).await?;

    // Step 3: 处理支付
    let payment = payment_service.process_payment(&order_request.payment).await?;

    // Step 4: 创建订单
    Ok(Order::create(user, products, payment))
}

// 如果函数超过 150 行，拆分为更小的函数
async fn validate_order_items(items: &[OrderItem]) -> anyhow::Result<Vec<Product>> {
    // 专门的验证逻辑
}

async fn calculate_order_total(items: &[ValidatedItem]) -> anyhow::Result<Money> {
    // 专门的计算逻辑
}
```

**文件大小控制 (≤ 500 行)：**
```rust
// src/users/mod.rs - 模块组织
pub mod model;
pub mod service;
pub mod repository;
pub mod handlers;
pub mod validation;

pub use model::{User, NewUser, UpdateUser};
pub use service::UserService;
pub use repository::UserRepository;
```

## 🔧 类型系统最佳实践

### 1. Newtype 模式应用

**强类型设计：**
```rust
// ✅ 使用 newtype 增强类型安全
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UserId(pub uuid::Uuid);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmailAddress(String);

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Price(rust_decimal::Decimal);

impl Price {
    pub fn new(value: rust_decimal::Decimal) -> anyhow::Result<Self> {
        if value >= rust_decimal::Decimal::ZERO {
            Ok(Price(value))
        } else {
            Err(anyhow::anyhow!("Price cannot be negative"))
        }
    }

    pub fn as_decimal(&self) -> rust_decimal::Decimal {
        self.0
    }
}

// 防止类型混用
fn calculate_discount(price: Price, user_id: UserId) -> Price {
    // 编译器确保不会传错参数类型
}
```

### 2. Phantom Types 使用

**状态机模式：**
```rust
use std::marker::PhantomData;

pub struct Order<State> {
    id: uuid::Uuid,
    items: Vec<OrderItem>,
    total: Price,
    _state: PhantomData<State>,
}

pub struct Draft;
pub struct Confirmed;
pub struct Shipped;
pub struct Delivered;

impl Order<Draft> {
    pub fn new(items: Vec<OrderItem>) -> Self {
        Self {
            id: uuid::Uuid::new_v4(),
            items,
            total: Price::calculate_total(&items),
            _state: PhantomData,
        }
    }

    pub fn confirm(self) -> anyhow::Result<Order<Confirmed>> {
        // 验证订单
        if self.items.is_empty() {
            return Err(anyhow::anyhow!("Cannot confirm empty order"));
        }

        Ok(Order {
            id: self.id,
            items: self.items,
            total: self.total,
            _state: PhantomData,
        })
    }
}

impl Order<Confirmed> {
    pub fn ship(self) -> Order<Shipped> {
        Order {
            id: self.id,
            items: self.items,
            total: self.total,
            _state: PhantomData,
        }
    }
}

// 编译时确保状态转换正确
// let order = Order::new(items).confirm()?.ship();
```

### 3. 泛型和 Trait 设计

**灵活的抽象设计：**
```rust
// ✅ 使用 trait 定义行为抽象
#[async_trait::async_trait]
pub trait Repository<T, K> {
    type Error: std::error::Error + Send + Sync + 'static;

    async fn create(&self, entity: T) -> Result<T, Self::Error>;
    async fn find_by_id(&self, id: K) -> Result<Option<T>, Self::Error>;
    async fn update(&self, entity: T) -> Result<T, Self::Error>;
    async fn delete(&self, id: K) -> Result<(), Self::Error>;
}

// 具体实现
#[async_trait::async_trait]
impl Repository<User, UserId> for UserRepository {
    type Error = sqlx::Error;

    async fn create(&self, user: User) -> Result<User, Self::Error> {
        sqlx::query_as!(
            User,
            "INSERT INTO users (id, email, name) VALUES ($1, $2, $3) RETURNING *",
            user.id.0,
            user.email.0,
            user.name
        )
        .fetch_one(&self.pool)
        .await
    }
}
```

## 🚀 性能优化标准

### 1. 内存管理优化

**智能内存分配：**
```rust
use smallvec::{SmallVec, smallvec};
use tinyvec::TinyVec;

// ✅ 使用 SmallVec 优化小集合
type SmallStringVec = SmallVec<[String; 4]>;

pub struct UserProfile {
    // 大多数用户标签数量 < 4，避免堆分配
    tags: SmallStringVec,

    // 使用 Box 减少结构体大小
    metadata: Box<UserMetadata>,
}

impl UserProfile {
    pub fn add_tag(&mut self, tag: String) {
        self.tags.push(tag);
        // SmallVec 自动处理溢出到堆
    }
}

// ✅ 使用 Cow 优化字符串处理
use std::borrow::Cow;

pub fn normalize_name(name: &str) -> Cow<str> {
    if name.chars().all(|c| c.is_ascii_lowercase()) {
        Cow::Borrowed(name)  // 无需分配
    } else {
        Cow::Owned(name.to_lowercase())  // 需要时才分配
    }
}
```

### 2. 无锁并发编程

**高性能并发原语：**
```rust
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use arc_swap::{ArcSwap, Guard};

pub struct MetricsCollector {
    // ❌ 不使用 Arc<Mutex<HashMap>>
    // counters: Arc<Mutex<HashMap<String, u64>>>,

    // ✅ 使用 DashMap 实现无锁并发
    counters: DashMap<String, AtomicU64>,

    // ✅ 使用原子计数器
    total_requests: AtomicU64,

    // ✅ 使用 ArcSwap 实现配置热更新
    config: ArcSwap<MetricsConfig>,
}

impl MetricsCollector {
    pub fn increment_counter(&self, key: &str) {
        self.counters
            .entry(key.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn get_total_requests(&self) -> u64 {
        self.total_requests.load(Ordering::Relaxed)
    }

    pub fn update_config(&self, new_config: MetricsConfig) {
        self.config.store(Arc::new(new_config));
    }

    pub fn get_config(&self) -> Guard<Arc<MetricsConfig>> {
        self.config.load()
    }
}
```

### 3. I/O 优化模式

**批量操作和连接池：**
```rust
use sqlx::{PgPool, Postgres, Transaction};

pub struct BatchUserRepository {
    pool: PgPool,
}

impl BatchUserRepository {
    // ✅ 批量插入优化
    pub async fn create_users_batch(&self, users: Vec<NewUser>) -> sqlx::Result<Vec<User>> {
        let mut tx = self.pool.begin().await?;
        let mut created_users = Vec::with_capacity(users.len());

        // 使用事务批量处理
        for user in users {
            let created = sqlx::query_as!(
                User,
                "INSERT INTO users (email, name) VALUES ($1, $2) RETURNING *",
                user.email,
                user.name
            )
            .fetch_one(&mut *tx)
            .await?;

            created_users.push(created);
        }

        tx.commit().await?;
        Ok(created_users)
    }

    // ✅ 连接池复用
    pub async fn find_users_by_ids(&self, ids: &[UserId]) -> sqlx::Result<Vec<User>> {
        let id_strings: Vec<String> = ids.iter().map(|id| id.0.to_string()).collect();

        sqlx::query_as!(
            User,
            "SELECT * FROM users WHERE id = ANY($1)",
            &id_strings
        )
        .fetch_all(&self.pool)
        .await
    }
}
```

## 🔒 安全编程标准

### 1. 输入验证和清理

**全面的输入验证：**
```rust
use validator::{Validate, ValidationError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Validate)]
pub struct CreateUserRequest {
    #[validate(email, message = "Invalid email format")]
    pub email: String,

    #[validate(length(min = 2, max = 50, message = "Name must be 2-50 characters"))]
    pub name: String,

    #[validate(length(min = 8, message = "Password must be at least 8 characters"))]
    #[validate(custom = "validate_password_strength")]
    pub password: String,

    #[validate(range(min = 18, max = 120, message = "Age must be between 18 and 120"))]
    pub age: u8,
}

fn validate_password_strength(password: &str) -> Result<(), ValidationError> {
    let has_uppercase = password.chars().any(|c| c.is_uppercase());
    let has_lowercase = password.chars().any(|c| c.is_lowercase());
    let has_digit = password.chars().any(|c| c.is_digit(10));
    let has_special = password.chars().any(|c| "!@#$%^&*()".contains(c));

    if has_uppercase && has_lowercase && has_digit && has_special {
        Ok(())
    } else {
        Err(ValidationError::new("Password must contain uppercase, lowercase, digit, and special character"))
    }
}

// 使用示例
pub async fn create_user(request: CreateUserRequest) -> anyhow::Result<User> {
    // 验证输入
    request.validate()
        .map_err(|e| anyhow::anyhow!("Validation failed: {}", e))?;

    // 处理请求
    Ok(User::new(request.email, request.name))
}
```

### 2. 密钥管理和加密

**安全的密钥处理：**
```rust
use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier, password_hash::SaltString};
use ring::{aead, rand};

pub struct SecurityService {
    argon2: Argon2<'static>,
    encryption_key: aead::LessSafeKey,
}

impl SecurityService {
    pub fn new() -> anyhow::Result<Self> {
        // ✅ 安全的密钥生成
        let mut key_bytes = [0u8; 32];
        rand::SystemRandom::new().fill(&mut key_bytes)
            .map_err(|_| anyhow::anyhow!("Failed to generate encryption key"))?;

        let unbound_key = aead::UnboundKey::new(&aead::CHACHA20_POLY1305, &key_bytes)
            .map_err(|_| anyhow::anyhow!("Failed to create encryption key"))?;

        Ok(Self {
            argon2: Argon2::default(),
            encryption_key: aead::LessSafeKey::new(unbound_key),
        })
    }

    // ✅ 安全的密码哈希
    pub fn hash_password(&self, password: &str) -> anyhow::Result<String> {
        let salt = SaltString::generate(&mut rand::thread_rng());

        let password_hash = self.argon2
            .hash_password(password.as_bytes(), &salt)
            .map_err(|e| anyhow::anyhow!("Password hashing failed: {}", e))?;

        Ok(password_hash.to_string())
    }

    // ✅ 安全的密码验证
    pub fn verify_password(&self, password: &str, hash: &str) -> anyhow::Result<bool> {
        let parsed_hash = PasswordHash::new(hash)
            .map_err(|e| anyhow::anyhow!("Invalid password hash: {}", e))?;

        match self.argon2.verify_password(password.as_bytes(), &parsed_hash) {
            Ok(()) => Ok(true),
            Err(_) => Ok(false),
        }
    }
}
```

### 3. 安全的错误处理

**不泄露敏感信息：**
```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum UserError {
    #[error("Authentication failed")]
    AuthenticationFailed,  // 不暴露具体原因

    #[error("User not found")]
    NotFound,

    #[error("Invalid input provided")]
    InvalidInput,  // 不暴露具体字段

    #[error("Internal server error")]
    Internal,  // 不暴露内部错误详情
}

pub struct UserService {
    repository: Arc<UserRepository>,
}

impl UserService {
    pub async fn authenticate(&self, email: &str, password: &str) -> Result<User, UserError> {
        let user = self.repository
            .find_by_email(email)
            .await
            .map_err(|e| {
                // ✅ 记录详细错误，但不暴露给用户
                tracing::error!("Database error during authentication: {}", e);
                UserError::Internal
            })?
            .ok_or(UserError::AuthenticationFailed)?;  // 不区分用户不存在和密码错误

        if self.verify_password(password, &user.password_hash)? {
            Ok(user)
        } else {
            Err(UserError::AuthenticationFailed)  // 统一的错误消息
        }
    }
}
```

这些核心标准确保所有 Rust 项目都能保持一致的高质量代码实践，同时兼顾性能、安全性和可维护性。
