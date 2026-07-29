use chrono::{Datelike, Timelike, Utc};
use futures::StreamExt;
use rig::agent::MultiTurnStreamItem;
use rig::prelude::*;
use rig::providers::deepseek;
use rig::streaming::{StreamedAssistantContent, StreamingPrompt};
use rmcp::model::{ClientCapabilities, ClientInfo, Implementation};
use rmcp::transport::TokioChildProcess;
use rmcp::ServiceExt;
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use tauri::ipc::Channel;
use tokio::process::Command;

struct AppConfig {
    key_path: PathBuf,
    history_dir: PathBuf,
    default_directory: PathBuf,
}

#[derive(Default, Deserialize, Serialize)]
struct ApiKeys {
    deepseek_api_key: String,
    tavily_api_key: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct AgentRequest {
    deepseek_api_key: Option<String>,
    tavily_api_key: Option<String>,
    directory: String,
    prompt: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SettingsStatus {
    has_deepseek_key: bool,
    has_tavily_key: bool,
    default_directory: String,
}

#[derive(Clone, Serialize)]
#[serde(tag = "event", content = "data", rename_all = "camelCase")]
enum AgentEvent {
    Status(String),
    Output(String),
    Complete,
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let base_dir = std::env::current_dir().expect("failed to resolve current directory");
    let default_directory = base_dir.join("workspace");

    tauri::Builder::default()
        .manage(AppConfig {
            key_path: base_dir.join(".ai-keys.json"),
            history_dir: base_dir.join("chat"),
            default_directory,
        })
        .invoke_handler(tauri::generate_handler![get_settings, run_agent])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[tauri::command]
fn get_settings(config: tauri::State<'_, AppConfig>) -> Result<SettingsStatus, String> {
    let keys = load_keys(&config.key_path)?;

    Ok(SettingsStatus {
        has_deepseek_key: !keys.deepseek_api_key.is_empty(),
        has_tavily_key: !keys.tavily_api_key.is_empty(),
        default_directory: config.default_directory.to_string_lossy().into_owned(),
    })
}

#[tauri::command]
async fn run_agent(
    config: tauri::State<'_, AppConfig>,
    request: AgentRequest,
    on_event: Channel<AgentEvent>,
) -> Result<(), String> {
    let prompt = request.prompt.trim();
    if prompt.is_empty() {
        return Err("请输入任务内容".to_string());
    }

    let directory = PathBuf::from(request.directory.trim());
    let directory = directory
        .canonicalize()
        .map_err(|error| format!("无法访问指定目录: {error}"))?;
    if !directory.is_dir() {
        return Err("指定路径不是目录".to_string());
    }

    let keys = update_keys(
        &config.key_path,
        request.deepseek_api_key,
        request.tavily_api_key,
    )?;
    if keys.deepseek_api_key.is_empty() {
        return Err("请输入 DeepSeek API Key".to_string());
    }
    if keys.tavily_api_key.is_empty() {
        return Err("请输入 Tavily API Key".to_string());
    }

    send_event(
        &on_event,
        AgentEvent::Status("正在启动 MCP 服务".to_string()),
    )
    .map_err(|error| error.to_string())?;
    let response = execute_agent(prompt, &directory, &keys, &on_event)
        .await
        .map_err(|error| format!("AI 任务失败: {error:#}"))?;

    save_history(&config.history_dir, prompt, &response)
        .map_err(|error| format!("保存对话记录失败: {error}"))?;
    send_event(&on_event, AgentEvent::Complete).map_err(|error| error.to_string())?;
    Ok(())
}

async fn execute_agent(
    prompt: &str,
    directory: &Path,
    keys: &ApiKeys,
    on_event: &Channel<AgentEvent>,
) -> anyhow::Result<String> {
    let mut tavily_command = Command::new("npx");
    tavily_command
        .arg("-y")
        .arg("tavily-mcp@latest")
        .env("TAVILY_API_KEY", &keys.tavily_api_key);
    let (tavily_transport, _) = TokioChildProcess::builder(tavily_command).spawn()?;

    let mut filesystem_command = Command::new("npx");
    filesystem_command
        .arg("-y")
        .arg("@modelcontextprotocol/server-filesystem")
        .arg(directory);
    let (filesystem_transport, _) = TokioChildProcess::builder(filesystem_command).spawn()?;

    let client_info = ClientInfo::new(
        ClientCapabilities::default(),
        Implementation::new("local-tauri-agent".to_string(), "0.1.0".to_string()),
    );

    let tavily_service = client_info.clone().serve(tavily_transport).await?;
    let tavily_client = tavily_service.peer().clone();
    let tavily_tools = tavily_client.list_tools(Default::default()).await?.tools;

    let filesystem_service = client_info.serve(filesystem_transport).await?;
    let filesystem_client = filesystem_service.peer().clone();
    let filesystem_tools = filesystem_client
        .list_tools(Default::default())
        .await?
        .tools;

    send_event(on_event, AgentEvent::Status("AI 正在处理任务".to_string()))?;
    let client = deepseek::Client::builder()
        .api_key(&keys.deepseek_api_key)
        .build()?;
    let agent = client
        .agent(deepseek::DEEPSEEK_V4_FLASH)
        .preamble("你是一个有帮助的机器人。仅在用户指定的工作目录内读写文件。")
        .rmcp_tools(tavily_tools, tavily_client)
        .rmcp_tools(filesystem_tools, filesystem_client)
        .build();

    let mut stream = agent
        .stream_prompt(prompt)
        .max_turns(5)
        .max_tokens(393216)
        .await;
    let mut response = String::new();

    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(text)) =
            item?
        {
            response.push_str(&text.text);
            send_event(on_event, AgentEvent::Output(text.text))?;
        }
    }

    Ok(response)
}

fn load_keys(path: &Path) -> Result<ApiKeys, String> {
    if !path.exists() {
        return Ok(ApiKeys::default());
    }

    let content =
        fs::read_to_string(path).map_err(|error| format!("读取 API Key 失败: {error}"))?;
    serde_json::from_str(&content).map_err(|error| format!("API Key 文件格式错误: {error}"))
}

fn update_keys(
    path: &Path,
    deepseek_api_key: Option<String>,
    tavily_api_key: Option<String>,
) -> Result<ApiKeys, String> {
    let mut keys = load_keys(path)?;
    let mut changed = false;

    if let Some(key) = non_empty_key(deepseek_api_key) {
        keys.deepseek_api_key = key;
        changed = true;
    }
    if let Some(key) = non_empty_key(tavily_api_key) {
        keys.tavily_api_key = key;
        changed = true;
    }

    if changed {
        write_keys(path, &keys)?;
    }
    Ok(keys)
}

fn non_empty_key(key: Option<String>) -> Option<String> {
    key.map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn write_keys(path: &Path, keys: &ApiKeys) -> Result<(), String> {
    let mut options = OpenOptions::new();
    options.create(true).write(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }

    let file = options
        .open(path)
        .map_err(|error| format!("保存 API Key 失败: {error}"))?;
    serde_json::to_writer_pretty(file, keys).map_err(|error| format!("保存 API Key 失败: {error}"))
}

fn save_history(history_dir: &Path, prompt: &str, response: &str) -> std::io::Result<()> {
    fs::create_dir_all(history_dir)?;
    let now = Utc::now();
    let (_, year) = now.year_ce();
    let path = history_dir.join(format!(
        "history-{}-{:02}-{:02}.md",
        year,
        now.month(),
        now.day()
    ));
    let (is_pm, hour) = now.hour12();
    let mut output = OpenOptions::new().create(true).append(true).open(path)?;

    write!(
        output,
        "\n## Q: {prompt} {:02}:{:02}:{:02} {}\n\n{response}\n",
        hour,
        now.minute(),
        now.second(),
        if is_pm { "PM" } else { "AM" }
    )
}

fn send_event(channel: &Channel<AgentEvent>, event: AgentEvent) -> tauri::Result<()> {
    channel.send(event)
}
