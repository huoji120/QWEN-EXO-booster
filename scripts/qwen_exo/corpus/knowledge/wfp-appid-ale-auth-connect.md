---
canonical: true
quality: 1.0
source_kind: local_sdk_verified
sdk_version: 10.0.19041.0
scope: user-mode WFP AppID outbound block and restore
---

# WFP SDK 10.0.19041：按 AppID 阻断和恢复进程出站连接

本文是本地 WFP 知识库唯一的技术参考。事实优先级如下：

1. 本机 Windows SDK 10.0.19041.0 头文件。
2. 本机真实编译错误。
3. 本文中明确标记的工程建议。

目标是在用户态使用 WFP management API，按可执行文件完整路径生成 AppID，并阻断该应用发起的新的 IPv4 和 IPv6 出站连接。简单阻断不需要 callout driver。

## 绝对不要使用的错误写法

- 不要使用不存在于本机 SDK 的 `FWPM_CONDITION_ALE_APPID`；正确名称是 `FWPM_CONDITION_ALE_APP_ID`。
- 不要把 WFP management API 返回值声明为 `NTSTATUS`，也不要用 `NT_SUCCESS` 判断。它们返回 `DWORD`，成功条件是 `status == ERROR_SUCCESS`。
- 不要把 EXE 路径字符串直接填入 AppID 条件。必须先调用 `FwpmGetAppIdFromFileName0` 得到 `FWP_BYTE_BLOB`。
- 不要仅声明一个 provider GUID 就把它赋给 `filter.providerKey`。非空 provider key 必须引用已经创建的 provider。
- 不要直接把 `NULL` 当作枚举句柄传给 `FwpmFilterEnum0`。必须先创建枚举句柄。
- 不要把动态会话设计成跨进程持久化的 `--status` / `--unblock`。动态对象随创建它们的会话结束而清理。
- 不要持久化 `FwpmFilterAdd0` 返回的运行时 `UINT64 filterId` 作为跨重启对象身份。持久方案应使用固定 `filterKey` GUID。

## 构建要求

```cpp
#include <windows.h>
#include <fwpmu.h>

#pragma comment(lib, "fwpuclnt.lib")
```

WFP策略写操作通常需要管理员权限，并依赖 Base Filtering Engine 服务可用。

## 本机 SDK 的准确标识符

```text
FWPM_SESSION0
FWPM_SESSION_FLAG_DYNAMIC
FWPM_FILTER0
FWPM_FILTER_CONDITION0
FWPM_CONDITION_ALE_APP_ID
FWP_BYTE_BLOB
FWP_BYTE_BLOB_TYPE
FWP_MATCH_EQUAL
FWPM_LAYER_ALE_AUTH_CONNECT_V4
FWPM_LAYER_ALE_AUTH_CONNECT_V6
FWP_ACTION_BLOCK
FWPM_SUBLAYER_UNIVERSAL
FWPM_FILTER_FLAG_PERSISTENT
FWPM_SUBLAYER_FLAG_PERSISTENT
FWPM_PROVIDER_FLAG_PERSISTENT
```

本机证据位置：

```text
C:\Program Files (x86)\Windows Kits\10\Include\10.0.19041.0\um\fwpmu.h
C:\Program Files (x86)\Windows Kits\10\Include\10.0.19041.0\shared\fwpmtypes.h
C:\Program Files (x86)\Windows Kits\10\Include\10.0.19041.0\shared\fwptypes.h
```

`FWPM_CONDITION_ALE_APP_ID` 位于 `fwpmu.h`。`FWPM_FILTER0`、持久化 flags 和枚举模板位于 `fwpmtypes.h`。`FWP_BYTE_BLOB_TYPE`、`FWP_MATCH_EQUAL`、`FWP_ACTION_BLOCK` 和 `FWPM_DISPLAY_DATA0` 位于 `fwptypes.h`。

## API 返回值规则

以下 management API 均返回 `DWORD`：

```cpp
FwpmEngineOpen0
FwpmEngineClose0
FwpmGetAppIdFromFileName0
FwpmFilterAdd0
FwpmFilterDeleteById0
FwpmFilterDeleteByKey0
FwpmFilterGetByKey0
FwpmFilterCreateEnumHandle0
FwpmFilterEnum0
FwpmFilterDestroyEnumHandle0
FwpmProviderAdd0
FwpmSubLayerAdd0
```

正确检查方式：

```cpp
DWORD status = FwpmFilterAdd0(engine, &filter, nullptr, &filterId);
if (status != ERROR_SUCCESS) {
    fwprintf(stderr, L"FwpmFilterAdd0 failed: 0x%08lX (%lu)\n",
             status, status);
}
```

不要使用：

```cpp
NTSTATUS status = FwpmFilterAdd0(...);
if (!NT_SUCCESS(status)) { ... }
```

普通正数 Win32 错误可能被 `NT_SUCCESS` 错误地当作成功。

`FwpmFreeMemory0` 返回 `void`：

```cpp
FwpmFreeMemory0(reinterpret_cast<void**>(&pointer));
```

## AppID 条件的准确构造

AppID 是 WFP 分配的不透明 byte blob，不是路径字符串：

```cpp
FWP_BYTE_BLOB* appId = nullptr;
DWORD status = FwpmGetAppIdFromFileName0(exePath, &appId);
if (status != ERROR_SUCCESS || appId == nullptr) {
    // Report the returned status directly.
}

FWPM_FILTER_CONDITION0 condition{};
condition.fieldKey = FWPM_CONDITION_ALE_APP_ID;
condition.matchType = FWP_MATCH_EQUAL;
condition.conditionValue.type = FWP_BYTE_BLOB_TYPE;
condition.conditionValue.byteBlob = appId;
```

`FwpmFilterAdd0` 完成后，调用方可以释放 AppID blob：

```cpp
FwpmFreeMemory0(reinterpret_cast<void**>(&appId));
```

## `FWPM_FILTER0` 的关键字段类型

```text
filterKey          GUID
displayData        FWPM_DISPLAY_DATA0
flags              UINT32
providerKey        GUID*              nullable
providerData       FWP_BYTE_BLOB
layerKey           GUID
subLayerKey        GUID
weight             FWP_VALUE0
numFilterConditions UINT32
filterCondition    FWPM_FILTER_CONDITION0*
action             FWPM_ACTION0
reserved           GUID*
filterId           UINT64             runtime ID
effectiveWeight    FWP_VALUE0
```

重要区别：

- `providerKey` 是可空的 `GUID*`。
- `layerKey`、`subLayerKey` 和 `filterKey` 是内嵌 `GUID` 值。
- `displayData` 是内嵌结构，名称字段写作 `filter.displayData.name`。
- `filterId` 是运行时 ID；`filterKey` 才适合持久对象的稳定身份。

## 最小过滤器构造

以下 helper 适用于已经打开 engine、已经准备 AppID、并且 `subLayerKey` 指向有效 sublayer 的情况：

```cpp
static DWORD AddAppBlockFilter(
    HANDLE engine,
    const GUID& layerKey,
    const GUID& subLayerKey,
    FWP_BYTE_BLOB* appId,
    wchar_t* displayName,
    UINT64* filterId)
{
    FWPM_FILTER_CONDITION0 condition{};
    condition.fieldKey = FWPM_CONDITION_ALE_APP_ID;
    condition.matchType = FWP_MATCH_EQUAL;
    condition.conditionValue.type = FWP_BYTE_BLOB_TYPE;
    condition.conditionValue.byteBlob = appId;

    FWPM_FILTER0 filter{};
    filter.displayData.name = displayName;
    filter.layerKey = layerKey;
    filter.subLayerKey = subLayerKey;
    filter.action.type = FWP_ACTION_BLOCK;
    filter.weight.type = FWP_EMPTY;
    filter.numFilterConditions = 1;
    filter.filterCondition = &condition;

    return FwpmFilterAdd0(engine, &filter, nullptr, filterId);
}
```

需要分别添加两个 filter：

```cpp
UINT64 filterIdV4 = 0;
UINT64 filterIdV6 = 0;

DWORD statusV4 = AddAppBlockFilter(
    engine,
    FWPM_LAYER_ALE_AUTH_CONNECT_V4,
    subLayerKey,
    appId,
    const_cast<wchar_t*>(L"Block application IPv4 outbound"),
    &filterIdV4);

DWORD statusV6 = AddAppBlockFilter(
    engine,
    FWPM_LAYER_ALE_AUTH_CONNECT_V6,
    subLayerKey,
    appId,
    const_cast<wchar_t*>(L"Block application IPv6 outbound"),
    &filterIdV6);
```

只添加 V4 filter 不会覆盖 IPv6。ALE AUTH CONNECT规则用于新的出站连接授权，不应宣称它会强制断开已经建立的连接。

## Provider 和 Sublayer

### Provider

Provider 对 filter 是可选的，因为 `FWPM_FILTER0.providerKey` 是可空指针。

合法选择：

1. 最小临时方案令 `filter.providerKey = nullptr`。
2. 先调用 `FwpmProviderAdd0` 创建 provider，再令 `filter.providerKey = &providerGuid`。

仅在代码里声明 provider GUID 不会创建 provider。引用不存在的 provider可能导致添加 filter失败。

### Sublayer

每个 filter 的 `subLayerKey` 必须指向有效 sublayer。

合法选择：

1. 明确使用已存在的 `FWPM_SUBLAYER_UNIVERSAL`。
2. 调用 `FwpmSubLayerAdd0` 创建应用自有 sublayer。

推荐应用自有 sublayer，因为它能表达所有权、分组和清理边界。GUID应在开发时生成一次并固定，不要每次运行随机生成。

生命周期必须匹配：

```text
动态filter     -> 动态session内创建的sublayer，或已有系统sublayer
持久filter     -> 持久sublayer
持久filter+provider -> 持久provider
```

## 架构 A：临时阻断，同一进程恢复

适用语义：

```text
控制程序活着时阻断
Ctrl+C或程序退出时恢复
```

准确流程：

1. `FWPM_SESSION0 session{}; session.flags = FWPM_SESSION_FLAG_DYNAMIC;`
2. 调用 `FwpmEngineOpen0`，并在整个阻断期间保持同一 engine handle存活。
3. 开始写事务。
4. 创建动态应用 sublayer，或明确使用 `FWPM_SUBLAYER_UNIVERSAL`。
5. 调用 `FwpmGetAppIdFromFileName0`。
6. 添加 `FWPM_LAYER_ALE_AUTH_CONNECT_V4` block filter。
7. 添加 `FWPM_LAYER_ALE_AUTH_CONNECT_V6` block filter。
8. 提交事务，避免只成功添加一个地址族。
9. 释放 AppID blob。
10. 保存本进程中的两个运行时 filter ID。
11. 恢复时按 ID 删除 V6 和 V4 filter，然后删除自有 sublayer。
12. 即使显式清理失败，关闭 engine或进程退出也会清理动态对象。

动态会话的关键语义：

- 规则生命周期绑定创建它的 WFP engine session。
- 目标应用退出不会删除规则。
- 创建规则的控制程序退出或关闭 engine会删除动态规则。
- 独立启动的新 `--status` / `--unblock` 进程不拥有旧进程的内存 filter ID。
- 如果旧控制进程已经退出，动态规则通常也已经不存在。

因此动态模式不应宣传独立跨进程 `--status` / `--unblock`。

动态 engine打开示例：

```cpp
FWPM_SESSION0 session{};
session.flags = FWPM_SESSION_FLAG_DYNAMIC;

HANDLE engine = nullptr;
DWORD status = FwpmEngineOpen0(
    nullptr,
    RPC_C_AUTHN_WINNT,
    nullptr,
    &session,
    &engine);
```

## 架构 B：持久阻断，跨进程 status/unblock

适用语义：

```text
block命令退出后规则仍存在
另一个进程可以status和unblock
可选择跨重启保留
```

准确流程：

1. 为 provider、sublayer、V4 filter和V6 filter定义四个固定 GUID。
2. 使用普通非动态 engine session。
3. 开始事务。
4. 创建或确认持久 provider：`FWPM_PROVIDER_FLAG_PERSISTENT`。
5. 创建或确认持久 sublayer：`FWPM_SUBLAYER_FLAG_PERSISTENT`。
6. 获取 AppID blob。
7. 添加固定 `filterKey` 的 V4 filter，设置 `FWPM_FILTER_FLAG_PERSISTENT`。
8. 添加固定 `filterKey` 的 V6 filter，设置 `FWPM_FILTER_FLAG_PERSISTENT`。
9. 提交事务并关闭 engine；规则继续存在。
10. `status` 使用 `FwpmFilterGetByKey0` 查询两个固定 filter key。
11. `unblock` 在事务中使用 `FwpmFilterDeleteByKey0` 删除两个固定 filter key。
12. 单独的 uninstall流程按 filter、sublayer、provider顺序删除对象。

持久模式不应保存运行时 filter ID作为长期身份。使用固定 `filterKey`。

`status` 不应只检查对象是否存在，还应核对：

```text
providerKey
subLayerKey
layerKey
action.type == FWP_ACTION_BLOCK
FWPM_CONDITION_ALE_APP_ID
FWP_BYTE_BLOB_TYPE
FWPM_FILTER_FLAG_PERSISTENT
```

## 删除和状态查询

同一动态进程内按运行时 ID删除：

```cpp
if (filterIdV6 != 0) {
    DWORD status = FwpmFilterDeleteById0(engine, filterIdV6);
}
if (filterIdV4 != 0) {
    DWORD status = FwpmFilterDeleteById0(engine, filterIdV4);
}
```

持久对象按固定 key查询和删除：

```cpp
FWPM_FILTER0* filter = nullptr;
DWORD status = FwpmFilterGetByKey0(engine, &FILTER_V4_KEY, &filter);
if (status == ERROR_SUCCESS) {
    // Validate the returned object.
    FwpmFreeMemory0(reinterpret_cast<void**>(&filter));
}

status = FwpmFilterDeleteByKey0(engine, &FILTER_V4_KEY);
```

## 正确枚举流程

如果确实需要按 provider/layer等条件枚举，不要跳过枚举句柄：

```cpp
FWPM_FILTER_ENUM_TEMPLATE0 query{};
query.providerKey = &providerGuid;

HANDLE enumHandle = nullptr;
DWORD status = FwpmFilterCreateEnumHandle0(engine, &query, &enumHandle);
if (status != ERROR_SUCCESS) {
    // Report failure.
}

for (;;) {
    FWPM_FILTER0** entries = nullptr;
    UINT32 count = 0;

    status = FwpmFilterEnum0(
        engine,
        enumHandle,
        64,
        &entries,
        &count);

    if (status != ERROR_SUCCESS) {
        FwpmFreeMemory0(reinterpret_cast<void**>(&entries));
        break;
    }

    for (UINT32 i = 0; i < count; ++i) {
        FWPM_FILTER0* current = entries[i];
        if (current == nullptr) {
            continue;
        }
        if (current->providerKey != nullptr &&
            IsEqualGUID(*current->providerKey, providerGuid)) {
            // Process the matching filter.
        }
    }

    FwpmFreeMemory0(reinterpret_cast<void**>(&entries));
    if (count == 0) {
        break;
    }
}

FwpmFilterDestroyEnumHandle0(engine, enumHandle);
```

关键点：

- 枚举模板传给 `FwpmFilterCreateEnumHandle0`。
- 创建出的 enum handle传给每次 `FwpmFilterEnum0`。
- `entries` 的类型是 `FWPM_FILTER0**`，输出参数是 `FWPM_FILTER0***`。
- 每批 entries使用 `FwpmFreeMemory0` 释放。
- 完成后调用 `FwpmFilterDestroyEnumHandle0`。
- 解引用 `providerKey` 前必须检查它不是 `nullptr`。
- 只有两个已知固定 filter时，`FwpmFilterGetByKey0` 通常比全局枚举简单可靠。

## 事务和失败回滚

V4、V6及其依赖对象应在同一事务中操作：

```text
FwpmTransactionBegin0
-> Add/Create V4 and V6 objects
-> FwpmTransactionCommit0
```

任一步失败则：

```text
FwpmTransactionAbort0
```

这样可以避免只阻断 IPv4或只阻断 IPv6的部分状态。

## 内存和所有权

- `FwpmGetAppIdFromFileName0` 返回的 AppID由调用方通过 `FwpmFreeMemory0` 释放。
- `FwpmFilterGetByKey0` / `FwpmFilterGetById0` 返回的对象由调用方通过 `FwpmFreeMemory0` 释放。
- `FwpmFilterEnum0` 返回的 entries批次由调用方通过 `FwpmFreeMemory0` 释放。
- `FwpmFilterAdd0` 会复制/封送输入 filter；调用完成后可释放 AppID blob。
- engine handle由调用方通过 `FwpmEngineClose0` 关闭。
- 动态对象由创建它们的动态 session最终兜底清理。

## 最终实施检查表

### 临时模式

- 使用 `FWPM_SESSION_FLAG_DYNAMIC`。
- 使用同一长生命周期 engine完成阻断和恢复。
- provider为空，或先真正创建 provider。
- 使用有效 sublayer。
- 使用事务原子添加 V4和V6。
- 条件使用 `FWPM_CONDITION_ALE_APP_ID`。
- 条件值使用 `FWP_BYTE_BLOB_TYPE`。
- API状态使用 `DWORD`和`ERROR_SUCCESS`。
- 退出时按 ID显式删除，并以关闭动态 session兜底。
- 不提供误导性的跨进程持久 `--status` / `--unblock`。

### 持久模式

- 不使用动态 session。
- provider、sublayer、两个 filter使用固定 GUID。
- 依赖对象和 filter均设置相应 persistent flags。
- 使用固定 filter key实现 status和unblock。
- 用事务保证 V4/V6状态一致。
- status校验对象内容，不只校验存在性。
- uninstall按 filter、sublayer、provider顺序执行。

## 范围边界

本文仅覆盖用户态 management API按 AppID阻断新的 IPv4/IPv6出站连接。本文不承诺：

- 强制终止已经建立的连接。
- 处理入站连接。
- 修改或代理数据包。
- 解决与 Windows防火墙、EDR、VPN等其他 WFP provider之间的全部仲裁问题。
- 给出生产环境ACL和安全描述符策略。

这些需求需要独立设计和针对目标Windows环境验证，不能通过猜测字段或随机 GUID补齐。
